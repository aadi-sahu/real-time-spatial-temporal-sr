"""
Milestone 7: Export Real-ESRGAN to ONNX and run GPU inference via DirectML.
Goal: Full-resolution AI upscaling that actually uses your RX 6600 GPU,
replacing the slow CPU-only RealESRGANer wrapper from Milestone 6.
"""

import os
import time
import cv2
import torch
import numpy as np
import onnxruntime as ort
from basicsr.archs.rrdbnet_arch import RRDBNet

torch.set_num_threads(2)  # keep system responsive during export step


def load_pytorch_model(model_name: str, weights_path: str) -> torch.nn.Module:
    """Load RRDBNet architecture and pretrained weights (same as Milestone 6)."""
    if model_name == "general":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                         num_block=23, num_grow_ch=32, scale=4)
    elif model_name == "anime":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                         num_block=6, num_grow_ch=32, scale=4)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    state_dict = torch.load(weights_path, map_location="cpu")
    # Real-ESRGAN checkpoints store weights under a 'params_ema' or 'params' key
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def export_model_to_onnx(model: torch.nn.Module, onnx_path: str, tile_size: int = 256) -> None:
    """Export RRDBNet to ONNX with a fixed dummy tile size but dynamic axes."""
    if os.path.exists(onnx_path):
        print(f"ONNX file already exists, skipping export: {onnx_path}")
        return

    dummy_input = torch.randn(1, 3, tile_size, tile_size)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
        opset_version=17,
    )
    print(f"Exported: {onnx_path}")


def create_directml_session(onnx_path: str) -> ort.InferenceSession:
    """Create an ONNX Runtime session using DirectML (GPU) execution provider."""
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"Session using: {session.get_providers()}")
    return session


def preprocess_tile(tile_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (H,W,C) -> RGB float32 (1,C,H,W), matching Milestone 4's conversion."""
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    float_img = rgb.astype(np.float32) / 255.0
    chw = np.transpose(float_img, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def postprocess_tile(output: np.ndarray) -> np.ndarray:
    """RGB float32 (1,C,H,W) -> BGR uint8 (H,W,C)."""
    output = np.squeeze(output, axis=0)
    hwc = np.transpose(output, (1, 2, 0))
    uint8_img = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(uint8_img, cv2.COLOR_RGB2BGR)


def upscale_with_tiling(session: ort.InferenceSession, img_bgr: np.ndarray,
                         tile_size: int = 256, scale: int = 4) -> np.ndarray:
    """
    Split image into tiles, run each through the ONNX/DirectML session,
    stitch results into the final upscaled image.
    """
    h, w, _ = img_bgr.shape
    output_h, output_w = h * scale, w * scale
    result = np.zeros((output_h, output_w, 3), dtype=np.uint8)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    num_tiles = 0
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img_bgr[y:y + tile_size, x:x + tile_size]
            tile_h, tile_w, _ = tile.shape

            input_tensor = preprocess_tile(tile)
            output = session.run([output_name], {input_name: input_tensor})[0]
            upscaled_tile = postprocess_tile(output)

            out_y, out_x = y * scale, x * scale
            out_h, out_w = tile_h * scale, tile_w * scale
            result[out_y:out_y + out_h, out_x:out_x + out_w] = upscaled_tile

            num_tiles += 1

    print(f"Processed {num_tiles} tiles")
    return result


def run_pipeline(model_name: str, weights_path: str, onnx_path: str,
                  input_image_path: str, output_image_path: str) -> None:
    print(f"\n=== {model_name.upper()} model: PyTorch -> ONNX -> DirectML ===")

    pytorch_model = load_pytorch_model(model_name, weights_path)
    export_model_to_onnx(pytorch_model, onnx_path)
    session = create_directml_session(onnx_path)

    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load: {input_image_path}")
    print(f"Input shape: {img.shape}")

    start = time.time()
    output = upscale_with_tiling(session, img, tile_size=256, scale=4)
    elapsed = time.time() - start

    print(f"Output shape: {output.shape}")
    print(f"Inference time: {elapsed:.2f} seconds")

    success = cv2.imwrite(output_image_path, output)
    if not success:
        raise IOError(f"Failed to save: {output_image_path}")
    print(f"Saved: {output_image_path}")


if __name__ == "__main__":
    os.makedirs("models/onnx", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    input_image = "assets/test.jpg"  # full resolution this time

    run_pipeline(
        model_name="anime",
        weights_path="models/weights/RealESRGAN_x4plus_anime_6B.pth",
        onnx_path="models/onnx/anime_x4.onnx",
        input_image_path=input_image,
        output_image_path="outputs/test_anime_x4_gpu.png",
    )

    run_pipeline(
        model_name="general",
        weights_path="models/weights/RealESRGAN_x4plus.pth",
        onnx_path="models/onnx/general_x4.onnx",
        input_image_path=input_image,
        output_image_path="outputs/test_general_x4_gpu.png",
    )