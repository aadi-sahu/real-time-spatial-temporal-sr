"""
RetroClarity — Real-Time Spatial Super-Resolution for Retro Game Emulation
using Real-ESRGAN and DirectML.

A GPU-accelerated pipeline that captures live frames from any window
(emulator, game, or application), enhances them using a pretrained
Real-ESRGAN model via ONNX Runtime + DirectML, and displays the result
alongside the raw feed in real time.

Modes:
- Live Mode: continuous capture + enhancement, optimized for responsiveness
  (native-resolution capture, optional frame-skip on static scenes)
- Snapshot Mode: single-frame, full native-resolution enhancement using
  overlapping tiled inference with seam blending, for maximum quality

B.Tech Minor Project — 7th Semester
"""

import ctypes
try:
    ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

import os
import sys
import time
import threading
import traceback
import cv2
import numpy as np
import mss
import pygetwindow as gw
import onnxruntime as ort

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGridLayout, QComboBox, QSizePolicy, QCheckBox
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, Qt


# ============================ CONFIG ============================
APP_NAME = "RetroClarity"
TITLE_BAR_HEIGHT = 32
NATIVE_WIDTH = 320
NATIVE_HEIGHT = 240
FRAME_SKIP_THRESHOLD = 1.5
OUTPUT_SCALE = 2

MODEL_PATHS = {
    "Anime / Pixel Art (fast)": "models/onnx/anime_x4.onnx",
    "General / 3D (higher detail)": "models/onnx/general_x4.onnx",
}
# ==================================================================


# ---------------------------- Core helpers ----------------------------

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


def resize_keep_aspect(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize preserving aspect ratio, padding with black to avoid distortion."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off, x_off = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


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
    return cv2.absdiff(frame1, frame2).mean() < threshold


def build_session_options() -> ort.SessionOptions:
    """Tune ONNX Runtime for lowest latency. Does not alter model output quality."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    return opts


def make_blend_mask(th: int, tw: int) -> np.ndarray:
    y = np.linspace(-1, 1, th)
    x = np.linspace(-1, 1, tw)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    mask = (1 - np.abs(yy)) * (1 - np.abs(xx))
    return np.clip(mask, 0.05, 1.0)[:, :, None]


def snapshot_enhance(frame_bgr: np.ndarray, session: ort.InferenceSession,
                      tile_size: int = 256, overlap: int = 32) -> np.ndarray:
    """Full-quality enhancement: overlapping tiles with blended seams, full resolution."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    h, w, _ = frame_bgr.shape
    scale = 4
    stride = max(1, tile_size - overlap)

    result = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    weight = np.zeros((h * scale, w * scale, 1), dtype=np.float32)

    y = 0
    while y < h:
        x = 0
        while x < w:
            y_end, x_end = min(y + tile_size, h), min(x + tile_size, w)
            tile = frame_bgr[y:y_end, x:x_end]
            th, tw, _ = tile.shape

            input_tensor = preprocess(tile)
            output = session.run([output_name], {input_name: input_tensor})[0]
            upscaled_tile = postprocess(output).astype(np.float32)

            blend_mask = make_blend_mask(th * scale, tw * scale)
            oy, ox = y * scale, x * scale
            result[oy:oy + th * scale, ox:ox + tw * scale] += upscaled_tile * blend_mask
            weight[oy:oy + th * scale, ox:ox + tw * scale] += blend_mask

            x += stride
        y += stride

    weight = np.maximum(weight, 1e-6)
    return np.clip(result / weight, 0, 255).astype(np.uint8)


# ---------------------------- Shared state & threads ----------------------------

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.raw_frame = None
        self.enhanced_frame = None
        self.capture_fps = 0.0
        self.inference_fps = 0.0
        self.frames_skipped = 0
        self.running = True
        self.status = "Idle"


def capture_worker(state: SharedState, window: gw.Win32Window, title_bar_height: int) -> None:
    state.status = f"Locked onto: {window.title}"
    with mss.mss() as sct:
        frame_count, start_time = 0, time.time()
        while state.running:
            try:
                if window.isMinimized or window.width <= 0 or window.height <= 0:
                    state.status = "Target window minimized — showing last frame"
                    time.sleep(0.3)
                    continue

                region = get_capture_region(window, title_bar_height)
                if region["width"] <= 0 or region["height"] <= 0:
                    state.status = "Target window minimized — showing last frame"
                    time.sleep(0.3)
                    continue

                raw = np.array(sct.grab(region))
                frame_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                with state.lock:
                    state.raw_frame = frame_bgr
                    state.status = f"Locked onto: {window.title}"

                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    state.capture_fps = frame_count / elapsed
                    frame_count, start_time = 0, time.time()
            except Exception as e:
                state.status = f"Capture error: {e}"
                time.sleep(0.5)


def inference_worker(state: SharedState, session: ort.InferenceSession,
                      native_w: int, native_h: int, skip_threshold: float,
                      output_scale: int, skip_enabled: bool) -> None:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    frame_count, start_time = 0, time.time()
    last_processed = None

    while state.running:
        try:
            with state.lock:
                frame = state.raw_frame.copy() if state.raw_frame is not None else None
            if frame is None:
                time.sleep(0.03)
                continue

            frame = resize_keep_aspect(frame, native_w, native_h)

            if skip_enabled and last_processed is not None and \
               frames_are_similar(frame, last_processed, skip_threshold):
                with state.lock:
                    state.frames_skipped += 1
                time.sleep(0.01)
                continue

            last_processed = frame.copy()
            input_tensor = preprocess(frame)
            output = session.run([output_name], {input_name: input_tensor})[0]
            enhanced = postprocess(output)

            if output_scale != 4:
                target_h, target_w = int(native_h * output_scale), int(native_w * output_scale)
                enhanced = cv2.resize(enhanced, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            with state.lock:
                state.enhanced_frame = enhanced

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                state.inference_fps = frame_count / elapsed
                frame_count, start_time = 0, time.time()
        except Exception as e:
            state.status = f"Inference error: {e}"
            time.sleep(0.5)


# ---------------------------- UI widgets ----------------------------

def cv_to_qpixmap(frame_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class VideoLabel(QLabel):
    def __init__(self, placeholder_text: str):
        super().__init__(placeholder_text)
        self.setStyleSheet("background-color: black; color: #888;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 150)
        self._pixmap = None

    def set_frame(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self):
        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — Real-Time Spatial Super-Resolution")
        self.resize(1080, 720)

        self.state = SharedState()
        self.session = None
        self.current_model_path = MODEL_PATHS["Anime / Pixel Art (fast)"]
        self.cap_thread = None
        self.inf_thread = None

        self._build_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frames)
        self.timer.start(33)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        header_layout = QHBoxLayout()
        title = QLabel(APP_NAME)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header_layout.addWidget(title)
        header_layout.addStretch()
        self.stats_checkbox = QCheckBox("Stats")
        self.stats_checkbox.setChecked(False)
        self.stats_checkbox.stateChanged.connect(self._toggle_stats)
        header_layout.addWidget(self.stats_checkbox)
        layout.addLayout(header_layout)

        controls_layout = QGridLayout()

        self.window_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_window_list)
        controls_layout.addWidget(QLabel("Target Window:"), 0, 0)
        controls_layout.addWidget(self.window_combo, 0, 1)
        controls_layout.addWidget(self.refresh_btn, 0, 2)

        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_PATHS.keys()))
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        controls_layout.addWidget(QLabel("AI Model:"), 1, 0)
        controls_layout.addWidget(self.model_combo, 1, 1)

        self.frame_skip_checkbox = QCheckBox("Enable Frame Skip (faster on static scenes)")
        self.frame_skip_checkbox.setChecked(True)
        controls_layout.addWidget(self.frame_skip_checkbox, 1, 2)

        self.refresh_window_list()
        layout.addLayout(controls_layout)

        video_layout = QHBoxLayout()
        self.raw_label = VideoLabel("Raw feed")
        self.enhanced_label = VideoLabel("AI Enhanced feed")
        video_layout.addWidget(self.raw_label)
        video_layout.addWidget(self.enhanced_label)
        layout.addLayout(video_layout, stretch=1)

        self.stats_widget = QWidget()
        stats_layout = QGridLayout(self.stats_widget)
        self.raw_fps_label = QLabel("Raw FPS: --")
        self.ai_fps_label = QLabel("AI Enhanced FPS: --")
        self.skipped_label = QLabel("Frames skipped: --")
        for lbl in (self.raw_fps_label, self.ai_fps_label, self.skipped_label):
            lbl.setFont(QFont("Segoe UI", 9))
        stats_layout.addWidget(self.raw_fps_label, 0, 0)
        stats_layout.addWidget(self.ai_fps_label, 0, 1)
        stats_layout.addWidget(self.skipped_label, 0, 2)
        self.stats_widget.setVisible(False)
        layout.addWidget(self.stats_widget)

        self.status_label = QLabel("Status: idle")
        self.status_label.setFont(QFont("Segoe UI", 9))
        layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Live Mode")
        self.stop_btn = QPushButton("Stop")
        self.snapshot_btn = QPushButton("Snapshot Enhance (Full Quality)")
        self.fullscreen_btn = QPushButton("Fullscreen")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_pipeline)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        for b in (self.start_btn, self.stop_btn, self.snapshot_btn, self.fullscreen_btn):
            button_layout.addWidget(b)
        layout.addLayout(button_layout)

    def _toggle_stats(self):
        self.stats_widget.setVisible(self.stats_checkbox.isChecked())

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def refresh_window_list(self):
        self.window_combo.clear()
        self.window_combo.addItems([t for t in gw.getAllTitles() if t.strip()])

    def on_model_changed(self):
        selected = self.model_combo.currentText()
        new_path = MODEL_PATHS.get(selected)
        if new_path and new_path != self.current_model_path:
            self.current_model_path = new_path
            self.session = None
            self.status_label.setText(f"Status: model set to '{selected}'")

    def _ensure_session(self):
        if self.session is None:
            if not os.path.exists(self.current_model_path):
                raise FileNotFoundError(f"Model not found: {self.current_model_path}")
            self.session = ort.InferenceSession(
                self.current_model_path,
                sess_options=build_session_options(),
                providers=["DmlExecutionProvider", "CPUExecutionProvider"]
            )

    def start_pipeline(self):
        selected_title = self.window_combo.currentText()
        if not selected_title:
            self.status_label.setText("Status: no window selected")
            return
        try:
            window = find_window(selected_title)
        except RuntimeError as e:
            self.status_label.setText(f"Status: {e}")
            return

        self.status_label.setText("Status: loading model...")
        QApplication.processEvents()
        try:
            self._ensure_session()
        except Exception as e:
            traceback.print_exc()
            self.status_label.setText(f"Status: model load error - {e}")
            return

        self.state = SharedState()
        self.cap_thread = threading.Thread(
            target=capture_worker, args=(self.state, window, TITLE_BAR_HEIGHT), daemon=True
        )
        self.inf_thread = threading.Thread(
            target=inference_worker,
            args=(self.state, self.session, NATIVE_WIDTH, NATIVE_HEIGHT, FRAME_SKIP_THRESHOLD,
                  OUTPUT_SCALE, self.frame_skip_checkbox.isChecked()),
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

    def take_snapshot(self):
        with self.state.lock:
            raw = self.state.raw_frame.copy() if self.state.raw_frame is not None else None
        if raw is None:
            self.status_label.setText("Status: no frame available — start live mode first")
            return

        self.status_label.setText("Status: running full-quality enhancement...")
        QApplication.processEvents()
        try:
            self._ensure_session()
            enhanced_full = snapshot_enhance(raw, self.session)
            os.makedirs("outputs", exist_ok=True)
            cv2.imwrite("outputs/snapshot_raw.png", raw)
            cv2.imwrite("outputs/snapshot_enhanced.png", enhanced_full)
            self.raw_label.set_frame(cv_to_qpixmap(raw))
            self.enhanced_label.set_frame(cv_to_qpixmap(enhanced_full))
            self.status_label.setText("Status: snapshot saved to outputs/")
        except Exception as e:
            traceback.print_exc()
            self.status_label.setText(f"Status: snapshot error - {e}")

    def _update_frames(self):
        with self.state.lock:
            raw = self.state.raw_frame.copy() if self.state.raw_frame is not None else None
            enhanced = self.state.enhanced_frame.copy() if self.state.enhanced_frame is not None else None
            cap_fps, inf_fps, skipped, status = (
                self.state.capture_fps, self.state.inference_fps,
                self.state.frames_skipped, self.state.status
            )

        if self.state.running:
            if raw is not None:
                self.raw_label.set_frame(cv_to_qpixmap(raw))
            if enhanced is not None:
                self.enhanced_label.set_frame(cv_to_qpixmap(enhanced))

        if self.stats_checkbox.isChecked():
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