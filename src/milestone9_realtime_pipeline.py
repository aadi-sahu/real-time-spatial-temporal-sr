"""
Milestone 9: Real-time pipeline combining live emulator capture with
GPU-accelerated Real-ESRGAN enhancement, displayed side-by-side with FPS.
"""

import ctypes
try:
    ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

import time
import threading
import cv2
import numpy as np
import mss
import pygetwindow as gw
import onnxruntime as ort


# ---------- Window capture (from Milestone 8) ----------

def find_emulator_window(title_keyword: str) -> gw.Win32Window:
    windows = gw.getWindowsWithTitle(title_keyword)
    if not windows:
        raise RuntimeError(f"No window found with title containing '{title_keyword}'.")
    return windows[0]


def get_capture_region(window: gw.Win32Window, title_bar_height: int = 55) -> dict:
    return {
        "left": window.left,
        "top": window.top + title_bar_height,
        "width": window.width,
        "height": window.height - title_bar_height,
    }


# ---------- ONNX / DirectML inference (from Milestone 7) ----------

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


class SharedState:
    """Thread-safe container for the latest raw and enhanced frames."""
    def __init__(self):
        self.lock = threading.Lock()
        self.raw_frame: np.ndarray | None = None
        self.enhanced_frame: np.ndarray | None = None
        self.capture_fps: float = 0.0
        self.inference_fps: float = 0.0
        self.running = True


def capture_worker(state: SharedState, title_keyword: str) -> None:
    """Continuously capture frames as fast as possible."""
    with mss.mss() as sct:
        frame_count = 0
        start_time = time.time()

        while state.running:
            try:
                window = find_emulator_window(title_keyword)
                region = get_capture_region(window)
                if frame_count == 0:
                    print(f"[DEBUG] Capture region: {region['width']}x{region['height']}")
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

            except RuntimeError:
                time.sleep(0.5)  # emulator window not found, retry


def inference_worker(state: SharedState, session: ort.InferenceSession) -> None:
    """Continuously run AI enhancement on the latest available raw frame."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    frame_count = 0
    start_time = time.time()

    while state.running:
        with state.lock:
            frame_to_process = state.raw_frame.copy() if state.raw_frame is not None else None

        if frame_to_process is None:
            time.sleep(0.05)
            continue

        input_tensor = preprocess(frame_to_process)
        output = session.run([output_name], {input_name: input_tensor})[0]
        enhanced = postprocess(output)

        with state.lock:
            frame_to_process = state.raw_frame.copy() if state.raw_frame is not None else None
            if frame_to_process is not None:
                frame_to_process = cv2.resize(frame_to_process, (240, 160), interpolation=cv2.INTER_AREA)

        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            state.inference_fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()


def display_loop(state: SharedState) -> None:
    """Show raw and enhanced frames side by side with FPS overlays."""
    window_name = "Real-Time Super Resolution: Raw vs AI Enhanced"
    cv2.namedWindow(window_name)
    print("Press 'q' to stop.")

    while True:
        with state.lock:
            raw = state.raw_frame.copy() if state.raw_frame is not None else None
            enhanced = state.enhanced_frame.copy() if state.enhanced_frame is not None else None
            cap_fps = state.capture_fps
            inf_fps = state.inference_fps

        if raw is None:
            time.sleep(0.05)
            continue

        # Resize raw to match enhanced's display width for a clean side-by-side view,
        # or if enhanced isn't ready yet, just show raw twice as a placeholder.
        display_h = 480
        raw_resized = cv2.resize(raw, (int(raw.shape[1] * display_h / raw.shape[0]), display_h))

        if enhanced is not None:
            enhanced_resized = cv2.resize(
                enhanced, (int(enhanced.shape[1] * display_h / enhanced.shape[0]), display_h)
            )
        else:
            enhanced_resized = raw_resized.copy()

        cv2.putText(raw_resized, f"RAW  {cap_fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(enhanced_resized, f"AI ENHANCED  {inf_fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        combined = np.hstack([raw_resized, enhanced_resized])
        cv2.imshow(window_name, combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            state.running = False
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    onnx_path = "models/onnx/anime_x4.onnx"  # reuse the model exported in Milestone 7
    session = ort.InferenceSession(onnx_path, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    print(f"Inference session using: {session.get_providers()}")

    state = SharedState()

    cap_thread = threading.Thread(target=capture_worker, args=(state, "Pokemon - FireRed"), daemon=True)
    inf_thread = threading.Thread(target=inference_worker, args=(state, session), daemon=True)

    cap_thread.start()
    inf_thread.start()

    display_loop(state)