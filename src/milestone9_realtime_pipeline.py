"""
Milestone 9 (optimized): Real-time pipeline combining live window capture with
GPU-accelerated Real-ESRGAN enhancement, displayed side-by-side with FPS.

Optimizations added:
- Skip AI inference on frames that haven't changed (static screens, dialogue, menus)
- Downscale 4x model output to 2x for lighter downstream load
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


# ============ CONFIG — change these per game/emulator ============
WINDOW_TITLE_KEYWORD = "UNDERTALE"
TITLE_BAR_HEIGHT = 32          # crop this many px off the top (title bar)
NATIVE_WIDTH = 320             # game's native resolution, used before AI upscale
NATIVE_HEIGHT = 240
ONNX_MODEL_PATH = "models/onnx/anime_x4.onnx"
FRAME_SKIP_THRESHOLD = 2.0     # lower = more sensitive to change, higher = skips more
OUTPUT_SCALE = 2               # downscale model's native 4x output to this factor
# ===================================================================


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
    """Check if two frames are nearly identical (mean pixel difference below threshold)."""
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


def capture_worker(state: SharedState, title_keyword: str, title_bar_height: int) -> None:
    """Find the window ONCE, then continuously capture frames from it."""
    try:
        window = find_window(title_keyword)
        print(f"[CAPTURE] Locked onto window: '{window.title}' "
              f"at ({window.left},{window.top}) size {window.width}x{window.height}")
    except RuntimeError as e:
        print(f"[CAPTURE] FATAL - could not find window: {e}")
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
                print(f"[CAPTURE] Frame grab error: {e}")
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

            # Skip inference entirely if frame barely changed since last processed one
            if last_processed is not None and frames_are_similar(frame, last_processed, skip_threshold):
                with state.lock:
                    state.frames_skipped += 1
                time.sleep(0.02)
                continue

            last_processed = frame.copy()

            input_tensor = preprocess(frame)
            output = session.run([output_name], {input_name: input_tensor})[0]
            enhanced = postprocess(output)

            # Model outputs native 4x; downscale to desired output_scale (lighter, still enhanced)
            native_scale = 4
            if output_scale != native_scale:
                h, w = enhanced.shape[:2]
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
            print(f"[INFERENCE] Error: {e}")
            time.sleep(0.5)


def display_loop(state: SharedState) -> None:
    window_name = "Real-Time Super Resolution: Raw vs AI Enhanced"
    cv2.namedWindow(window_name)
    print("Press 'q' to stop.")

    while True:
        with state.lock:
            raw = state.raw_frame.copy() if state.raw_frame is not None else None
            enhanced = state.enhanced_frame.copy() if state.enhanced_frame is not None else None
            cap_fps = state.capture_fps
            inf_fps = state.inference_fps
            skipped = state.frames_skipped

        if raw is None:
            time.sleep(0.05)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                state.running = False
                break
            continue

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
        cv2.putText(enhanced_resized, f"AI ENHANCED  {inf_fps:.1f} FPS  (skipped: {skipped})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        combined = np.hstack([raw_resized, enhanced_resized])
        cv2.imshow(window_name, combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            state.running = False
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    session = ort.InferenceSession(
        ONNX_MODEL_PATH, providers=["DmlExecutionProvider", "CPUExecutionProvider"]
    )
    print(f"Inference session using: {session.get_providers()}")

    state = SharedState()

    cap_thread = threading.Thread(
        target=capture_worker, args=(state, WINDOW_TITLE_KEYWORD, TITLE_BAR_HEIGHT), daemon=True
    )
    inf_thread = threading.Thread(
        target=inference_worker,
        args=(state, session, NATIVE_WIDTH, NATIVE_HEIGHT, FRAME_SKIP_THRESHOLD, OUTPUT_SCALE),
        daemon=True
    )

    cap_thread.start()
    time.sleep(1)
    inf_thread.start()

    display_loop(state)