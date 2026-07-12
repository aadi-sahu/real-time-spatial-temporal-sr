"""
Milestone 5: ONNX & ONNX Runtime Basics
Goal: Build a tiny PyTorch model, export it to ONNX, then run inference
using ONNX Runtime with the DirectML execution provider (GPU on AMD).
"""

import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort


class TinyModel(nn.Module):
    """A trivial CNN: just doubles pixel intensity via a learned conv layer.
    Purpose: prove the export/inference pipeline works, not to be useful."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def export_to_onnx(model: nn.Module, onnx_path: str, input_shape: tuple) -> None:
    """Export a PyTorch model to ONNX format."""
    model.eval()  # inference mode: disables dropout/batchnorm training behavior
    dummy_input = torch.randn(input_shape)  # fake input just to trace the graph

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},   # allow variable image sizes
            "output": {2: "height", 3: "width"},
        },
        opset_version=17,
    )
    print(f"Exported model to: {onnx_path}")


def run_onnx_inference(onnx_path: str, input_array: np.ndarray) -> np.ndarray:
    """Load an ONNX model and run inference using DirectML (GPU) if available."""
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)

    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Session using: {session.get_providers()}")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: input_array})
    return result[0]


if __name__ == "__main__":
    model = TinyModel()
    onnx_path = "models_tiny_test.onnx"

    # Step 1: Export
    export_to_onnx(model, onnx_path, input_shape=(1, 3, 64, 64))

    # Step 2: Prepare a random input for inference (simulating a real frame)
    test_input = np.random.rand(1, 3, 64, 64).astype(np.float32)

    # Step 3: Run inference via ONNX Runtime
    output = run_onnx_inference(onnx_path, test_input)

    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Pipeline verified: PyTorch -> ONNX -> ONNX Runtime (DirectML) works.")