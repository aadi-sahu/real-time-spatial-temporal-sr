"""
Milestone 13: PyQt6 GUI wrapping the real-time capture + AI enhancement pipeline.
Reuses the exact same capture_worker / inference_worker logic from Milestone 9/11.
"""

import ctypes
try:
    ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

import sys
import time
import threading
import cv2
import numpy as np
import mss
import pygetwindow as gw
import onnxruntime as ort

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGridLayout
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt


# ============ CONFIG ============
WINDOW_TITLE_KEYWORD = "UNDERTALE"
TITLE_BAR_HEIGHT = 32
NATIVE_WIDTH = 320
NATIVE_HEIGHT = 240
ONNX_MODEL_PATH = "models/onnx/anime_x4.onnx"
FRAME_SKIP_THRESHOLD = 2.0
OUTPUT_SCALE = 2
# =================================


def find_window(title_keyword: str) -> gw.Win32Window:
    windows = gw.getWindowsWithTitle(title_keyword)
    if not windows:
        raise RuntimeError(f"No window found with title containing '{title_keyword}'.")
    return windows[0]


def get_capture_region(window: gw.Win32Window, title_bar_height: int) -> dict:
    return {
        "left": window.left,
        "top": window.top + title_bar_height,
        "width": window.width,
        "height": window.height - title_bar_height,
    }


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    float_img = rgb.astype(np.float32) / 255.0
    chw = np.transpose(float_img, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def postprocess(output: np.ndarray) -> np.ndarray:
    output = np.squeeze(output, axis=0)
    hwc = np.transpose(output, (1, 2, 0))
    uint8_img = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(uint8_img, cv2.COLOR_RGB2BGR)


def frames_are_similar(frame1: np.ndarray, frame2: np.ndarray, threshold: float) -> bool:
    diff = cv2.absdiff(frame1, frame2)
    return diff.mean() < threshold


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.raw_frame = None
        self.enhanced_frame = None
        self.capture_fps = 0.0
        self.inference_fps = 0.0
        self.frames_skipped = 0
        self.running = True
        self.status = "Starting..."


def capture_worker(state: SharedState, title_keyword: str, title_bar_height: int) -> None:
    try:
        window = find_window(title_keyword)
        state.status = f"Locked onto: {window.title}"
    except RuntimeError as e:
        state.status = f"ERROR: {e}"
        state.running = False
        return

    with mss.mss() as sct:
        frame_count = 0
        start_time = time.time()
        while state.running:
            try:
                region = get_capture_region(window, title_bar_height)
                raw = np.array(sct.grab(region))
                frame_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                with state.lock:
                    state.raw_frame = frame_bgr
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    state.capture_fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
            except Exception as e:
                state.status = f"Capture error: {e}"
                time.sleep(0.5)


def inference_worker(state: SharedState, session: ort.InferenceSession,
                      native_w: int, native_h: int, skip_threshold: float,
                      output_scale: int) -> None:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    frame_count = 0
    start_time = time.time()
    last_processed = None

    while state.running:
        try:
            with state.lock:
                frame = state.raw_frame.copy() if state.raw_frame is not None else None
            if frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (native_w, native_h), interpolation=cv2.INTER_AREA)

            if last_processed is not None and frames_are_similar(frame, last_processed, skip_threshold):
                with state.lock:
                    state.frames_skipped += 1
                time.sleep(0.02)
                continue

            last_processed = frame.copy()
            input_tensor = preprocess(frame)
            output = session.run([output_name], {input_name: input_tensor})[0]
            enhanced = postprocess(output)

            if output_scale != 4:
                target_h = int(native_h * output_scale)
                target_w = int(native_w * output_scale)
                enhanced = cv2.resize(enhanced, (target_w, target_h), interpolation=cv2.INTER_AREA)

            with state.lock:
                state.enhanced_frame = enhanced

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                state.inference_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
        except Exception as e:
            state.status = f"Inference error: {e}"
            time.sleep(0.5)


def cv_to_qpixmap(frame_bgr: np.ndarray, target_h: int = 380) -> QPixmap:
    """Convert an OpenCV BGR frame to a QPixmap for display in a QLabel."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    scale = target_h / h
    new_w, new_h = int(w * scale), target_h
    rgb = cv2.resize(rgb, (new_w, new_h))
    bytes_per_line = ch * new_w
    qimg = QImage(rgb.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Spatial Super Resolution — Minor Project")
        self.resize(1000, 600)

        self.state = SharedState()
        self.session = None
        self.cap_thread = None
        self.inf_thread = None

        self._build_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frames)
        self.timer.start(33)  # ~30 UI refreshes/sec (independent of AI FPS)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("Real-Time Spatial Super-Resolution — Retro Emulation")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        video_layout = QHBoxLayout()

        self.raw_label = QLabel("Raw feed will appear here")
        self.raw_label.setFixedSize(480, 380)
        self.raw_label.setStyleSheet("background-color: black; color: white;")
        self.raw_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.enhanced_label = QLabel("AI Enhanced feed will appear here")
        self.enhanced_label.setFixedSize(480, 380)
        self.enhanced_label.setStyleSheet("background-color: black; color: white;")
        self.enhanced_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        video_layout.addWidget(self.raw_label)
        video_layout.addWidget(self.enhanced_label)
        layout.addLayout(video_layout)

        stats_layout = QGridLayout()
        self.raw_fps_label = QLabel("Raw FPS: --")
        self.ai_fps_label = QLabel("AI Enhanced FPS: --")
        self.skipped_label = QLabel("Frames skipped: --")
        self.status_label = QLabel("Status: idle")
        for lbl in (self.raw_fps_label, self.ai_fps_label, self.skipped_label, self.status_label):
            lbl.setFont(QFont("Segoe UI", 10))
        stats_layout.addWidget(self.raw_fps_label, 0, 0)
        stats_layout.addWidget(self.ai_fps_label, 0, 1)
        stats_layout.addWidget(self.skipped_label, 1, 0)
        stats_layout.addWidget(self.status_label, 1, 1)
        layout.addLayout(stats_layout)

        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_pipeline)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)

    def start_pipeline(self):
        self.status_label.setText("Status: loading model...")
        self.session = ort.InferenceSession(
            ONNX_MODEL_PATH, providers=["DmlExecutionProvider", "CPUExecutionProvider"]
        )
        self.state = SharedState()

        self.cap_thread = threading.Thread(
            target=capture_worker, args=(self.state, WINDOW_TITLE_KEYWORD, TITLE_BAR_HEIGHT), daemon=True
        )
        self.inf_thread = threading.Thread(
            target=inference_worker,
            args=(self.state, self.session, NATIVE_WIDTH, NATIVE_HEIGHT, FRAME_SKIP_THRESHOLD, OUTPUT_SCALE),
            daemon=True
        )
        self.cap_thread.start()
        self.inf_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_pipeline(self):
        self.state.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: stopped")

    def _update_frames(self):
        with self.state.lock:
            raw = self.state.raw_frame.copy() if self.state.raw_frame is not None else None
            enhanced = self.state.enhanced_frame.copy() if self.state.enhanced_frame is not None else None
            cap_fps = self.state.capture_fps
            inf_fps = self.state.inference_fps
            skipped = self.state.frames_skipped
            status = self.state.status

        if raw is not None:
            self.raw_label.setPixmap(cv_to_qpixmap(raw))
        if enhanced is not None:
            self.enhanced_label.setPixmap(cv_to_qpixmap(enhanced))

        self.raw_fps_label.setText(f"Raw FPS: {cap_fps:.1f}")
        self.ai_fps_label.setText(f"AI Enhanced FPS: {inf_fps:.1f}")
        self.skipped_label.setText(f"Frames skipped: {skipped}")
        self.status_label.setText(f"Status: {status}")

    def closeEvent(self, event):
        self.state.running = False
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())