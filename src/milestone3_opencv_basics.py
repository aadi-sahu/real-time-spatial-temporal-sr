"""
Milestone 3: OpenCV Basics
Goal: Load an image, inspect it, convert color space, resize, and display it.
"""

import cv2
import numpy as np

def load_and_inspect(image_path: str) -> np.ndarray:
    """Load an image from disk and print its basic properties."""
    img = cv2.imread(image_path)  # Loaded as BGR, uint8

    if img is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    print(f"Shape: {img.shape}")       # (height, width, channels)
    print(f"Dtype: {img.dtype}")       # uint8
    print(f"Min/Max pixel value: {img.min()} / {img.max()}")

    return img


def convert_and_resize(img: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Convert BGR->RGB and resize the image by a scale factor."""
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)
    resized = cv2.resize(rgb_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    print(f"Resized shape: {resized.shape}")
    return resized


def display_image(img: np.ndarray, window_name: str = "Result") -> None:
    """Display an image in a window. Note: cv2.imshow expects BGR, so we convert back."""
    bgr_display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, bgr_display)
    print("Press any key on the image window to close it...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "assets/test.jpg"

    original = load_and_inspect(image_path)
    processed = convert_and_resize(original, scale=0.5)
    display_image(processed)