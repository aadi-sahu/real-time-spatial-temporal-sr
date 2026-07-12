"""
Milestone 4: PyTorch Basics
Goal: Understand tensors, and convert between OpenCV (NumPy/HWC/BGR/uint8)
and PyTorch (Tensor/CHW/RGB/float32) formats correctly.
"""

import cv2
import torch
import numpy as np


def image_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert an OpenCV BGR uint8 image (H, W, C) into a
    PyTorch RGB float32 tensor (C, H, W), normalized to [0, 1].
    """
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # uint8 [0,255] -> float32 [0,1]
    img_float = img_rgb.astype(np.float32) / 255.0

    # HWC -> CHW
    img_chw = np.transpose(img_float, (2, 0, 1))

    # NumPy -> Tensor
    tensor = torch.from_numpy(img_chw)

    # Add batch dimension: (C,H,W) -> (1,C,H,W)
    # Models expect a "batch" of images, even if batch size is 1
    tensor = tensor.unsqueeze(0)

    return tensor


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch RGB float32 tensor (1, C, H, W) back into
    an OpenCV-displayable BGR uint8 image (H, W, C).
    """
    # Remove batch dimension: (1,C,H,W) -> (C,H,W)
    tensor = tensor.squeeze(0)

    # CHW -> HWC
    img_chw_to_hwc = tensor.permute(1, 2, 0).numpy()

    # float32 [0,1] -> uint8 [0,255]
    img_uint8 = np.clip(img_chw_to_hwc * 255.0, 0, 255).astype(np.uint8)

    # RGB -> BGR
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    return img_bgr


if __name__ == "__main__":
    img = cv2.imread("assets/test.jpg")

    print("=== Original OpenCV image ===")
    print(f"Shape: {img.shape}, Dtype: {img.dtype}")

    tensor = image_to_tensor(img)
    print("\n=== Converted to PyTorch tensor ===")
    print(f"Shape: {tensor.shape}, Dtype: {tensor.dtype}")
    print(f"Min/Max value: {tensor.min().item():.3f} / {tensor.max().item():.3f}")

    recovered = tensor_to_image(tensor)
    print("\n=== Converted back to OpenCV image ===")
    print(f"Shape: {recovered.shape}, Dtype: {recovered.dtype}")

    cv2.imshow("Recovered Image (should look identical to original)", recovered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()