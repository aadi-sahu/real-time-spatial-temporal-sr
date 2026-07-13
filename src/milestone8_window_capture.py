"""
Milestone 8: Capture live frames from the mGBA emulator window.
Goal: Locate the emulator window, capture its contents continuously,
and display the raw captured frames in real time.
"""
import ctypes
try:
    ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)  # PER_MONITOR_AWARE_V2
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()  # fallback for older Windows

import cv2
import numpy as np
import mss
import pygetwindow as gw


def find_emulator_window(title_keyword: str = "mGBA") -> gw.Win32Window:
    """Find the mGBA window by matching part of its title."""
    windows = gw.getWindowsWithTitle(title_keyword)
    if not windows:
        raise RuntimeError(
            f"No window found with title containing '{title_keyword}'. "
            "Make sure mGBA is open and a ROM is loaded."
        )
    return windows[0]


def get_capture_region(window: gw.Win32Window, title_bar_height: int = 55) -> dict:
    """
    Convert a window's position/size into an mss-compatible capture region,
    cropping out the title bar and menu bar so we only capture the game screen.
    """
    return {
        "left": window.left,
        "top": window.top + title_bar_height,
        "width": window.width,
        "height": window.height - title_bar_height,
    }


import time

def capture_loop(title_keyword: str) -> None:
    """Continuously find the window, capture its contents, and display live."""
    with mss.mss() as sct:
        window_name = "Live Emulator Capture"
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 1200, 600)
        cv2.waitKey(500)

        print("Press 'q' in the display window to stop capturing.")

        while True:
            try:
                window = find_emulator_window(title_keyword)
            except RuntimeError:
                print("Emulator window not found (closed or minimized?). Stopping.")
                break

            region = get_capture_region(window)
            raw_frame = np.array(sct.grab(region))
            frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2BGR)
            cv2.imshow(window_name, frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_loop("Pokemon - FireRed")


if __name__ == "__main__":
    window = find_emulator_window("Pokemon - FireRed")
    print(f"Found window: '{window.title}' at "
          f"({window.left}, {window.top}), size {window.width}x{window.height}")

    region = get_capture_region(window)
    capture_loop(region)