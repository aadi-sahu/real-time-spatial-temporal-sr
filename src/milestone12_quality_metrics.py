"""
Milestone 12: Image Quality Metrics - PSNR, SSIM, LPIPS
Goal: Quantitatively compare Bicubic vs Real-ESRGAN upscaling against
a high-resolution ground truth, producing a results table for the report.
"""

import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def load_lpips_model() -> lpips.LPIPS:
    """Load the pretrained LPIPS perceptual similarity model (CPU, small, one-time load)."""
    return lpips.LPIPS(net='alex')  # AlexNet backbone - standard choice, lightweight


def image_to_lpips_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR uint8 image to the tensor format LPIPS expects: RGB, [-1, 1], CHW, batched."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    float_img = rgb.astype(np.float32) / 127.5 - 1.0  # normalize to [-1, 1]
    chw = np.transpose(float_img, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).float()
    return tensor


def compute_all_metrics(ground_truth: np.ndarray, candidate: np.ndarray,
                         lpips_model: lpips.LPIPS) -> dict:
    """
    Compute PSNR, SSIM, and LPIPS between a ground truth image and a candidate,
    resizing the candidate to match ground truth dimensions if needed.
    """
    if ground_truth.shape != candidate.shape:
        candidate = cv2.resize(candidate, (ground_truth.shape[1], ground_truth.shape[0]),
                                interpolation=cv2.INTER_CUBIC)

    psnr_value = compute_psnr(ground_truth, candidate, data_range=255)

    ssim_value = compute_ssim(ground_truth, candidate, channel_axis=2, data_range=255)

    gt_tensor = image_to_lpips_tensor(ground_truth)
    cand_tensor = image_to_lpips_tensor(candidate)
    with torch.no_grad():
        lpips_value = lpips_model(gt_tensor, cand_tensor).item()

    return {
        "PSNR (dB, higher=better)": round(psnr_value, 3),
        "SSIM (0-1, higher=better)": round(ssim_value, 4),
        "LPIPS (lower=better)": round(lpips_value, 4),
    }


def downscale_then_upscale(img: np.ndarray, factor: int = 4) -> np.ndarray:
    """
    Simulate a low-res source: shrink the ground truth down, then bicubic
    upscale it back up. This becomes our 'Bicubic' comparison candidate.
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return upscaled, small


if __name__ == "__main__":
    print("Loading LPIPS model (first run downloads pretrained weights)...")
    lpips_model = load_lpips_model()

    # Ground truth: your original high-res test image
    ground_truth = cv2.imread("assets/test.jpg")
    if ground_truth is None:
        raise FileNotFoundError("assets/test.jpg not found")

    print(f"Ground truth shape: {ground_truth.shape}")

    # Create the low-res source and bicubic baseline
    bicubic_result, low_res = downscale_then_upscale(ground_truth, factor=4)
    cv2.imwrite("outputs/metrics_low_res_source.png", low_res)
    cv2.imwrite("outputs/metrics_bicubic.png", bicubic_result)

    # Load the already-generated ESRGAN output from Milestone 7 for comparison
    esrgan_result = cv2.imread("outputs/test_anime_x4_gpu.png")
    if esrgan_result is None:
        raise FileNotFoundError(
            "outputs/test_anime_x4_gpu.png not found - run Milestone 7 first "
            "to generate the ESRGAN output for comparison."
        )

    print("\n=== Computing metrics: Bicubic vs Ground Truth ===")
    bicubic_metrics = compute_all_metrics(ground_truth, bicubic_result, lpips_model)
    for k, v in bicubic_metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Computing metrics: Real-ESRGAN vs Ground Truth ===")
    esrgan_metrics = compute_all_metrics(ground_truth, esrgan_result, lpips_model)
    for k, v in esrgan_metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Summary Table ===")
    print(f"{'Method':<12}{'PSNR':<10}{'SSIM':<10}{'LPIPS':<10}")
    print(f"{'Bicubic':<12}{bicubic_metrics['PSNR (dB, higher=better)']:<10}"
          f"{bicubic_metrics['SSIM (0-1, higher=better)']:<10}"
          f"{bicubic_metrics['LPIPS (lower=better)']:<10}")
    print(f"{'Real-ESRGAN':<12}{esrgan_metrics['PSNR (dB, higher=better)']:<10}"
          f"{esrgan_metrics['SSIM (0-1, higher=better)']:<10}"
          f"{esrgan_metrics['LPIPS (lower=better)']:<10}")