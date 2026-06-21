"""
Export FaceNet (InceptionResnetV1) from facenet-pytorch to ONNX format
for serving via NVIDIA Triton Inference Server.

Usage:
    # From project root (with projectmlops conda env active):
    python scripts/export_facenet_onnx.py

Output:
    models/facenet/1/model.onnx   (~100 MB)
"""
import os
import pathlib
import torch
from facenet_pytorch import InceptionResnetV1

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = pathlib.Path("models/facenet/1")
OUTPUT_PATH = OUTPUT_DIR / "model.onnx"

INPUT_SHAPE = (1, 3, 160, 160)  # (batch, channels, H, W)
OPSET_VERSION = 12               # Triton onnxruntime backend is compatible


def main():
    print("▶  Loading InceptionResnetV1 (vggface2)...")
    model = InceptionResnetV1(pretrained="vggface2").eval()

    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create a dummy input tensor
    dummy_input = torch.randn(*INPUT_SHAPE)

    print(f"▶  Exporting to ONNX → {OUTPUT_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_PATH),
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"✅  Export complete! File size: {size_mb:.1f} MB")
    print(f"    Path: {OUTPUT_PATH.resolve()}")
    print()
    print("Next steps:")
    print("  1. docker compose up -d triton")
    print("  2. curl http://localhost:8100/v2/health/ready")
    print("  3. docker compose up -d api  # will now use Triton path")


if __name__ == "__main__":
    main()
