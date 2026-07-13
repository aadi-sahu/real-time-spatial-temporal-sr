"""
Milestone 6: Load pretrained Real-ESRGAN models (general + anime)
Goal: Run real AI super-resolution on our test image using both models,
save outputs for visual comparison.
"""

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch
torch.set_num_threads(2)  # limit CPU usage so system stays responsive


def load_model(model_name: str, weights_path: str, device: str = "cpu") -> RealESRGANer:
    """
    Load a Real-ESRGAN model given its architecture config and weights.
    We run on CPU here since RealESRGANer's built-in device handling
    expects CUDA or CPU — DirectML integration comes in a later milestone
    once we export to ONNX. For now, CPU inference proves correctness.
    """
    if model_name == "general":
        # x4plus: standard RRDB architecture, 23 blocks
        arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
    elif model_name == "anime":
        # anime_6B: lighter architecture, 6 blocks, faster + tuned for flat colors
        arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=6, num_grow_ch=32, scale=4)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    upsampler = RealESRGANer(
        scale=4,
        model_path=weights_path,
        model=arch,
        tile=256,        # process in 256x256 tiles to avoid CPU memory overload
        tile_pad=10,
        pre_pad=0,
        half=False,      # FP16 disabled for now (CPU doesn't support it well)
        device=device,
    )
    return upsampler


def upscale_image(upsampler: RealESRGANer, img_path: str, output_path: str) -> None:
    """Run an image through the upsampler and save the result."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load: {img_path}")

    print(f"Input shape: {img.shape}")
    output, _ = upsampler.enhance(img, outscale=4)
    print(f"Output shape: {output.shape}")

    cv2.imwrite(output_path, output)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    input_image = "assets/test.jpg"

    small = cv2.imread(input_image)
    small = cv2.resize(small, (200, 100), interpolation=cv2.INTER_AREA)
    cv2.imwrite("assets/test_small.jpg", small)
    input_image = "assets/test_small.jpg"

    print("=== Loading GENERAL model (x4plus) ===")
    general_model = load_model("general", "models/weights/RealESRGAN_x4plus.pth")
    upscale_image(general_model, input_image, "outputs/test_general_x4.png")

    print("\n=== Loading ANIME model (x4plus_anime_6B) ===")
    anime_model = load_model("anime", "models/weights/RealESRGAN_x4plus_anime_6B.pth")
    upscale_image(anime_model, input_image, "outputs/test_anime_x4.png")