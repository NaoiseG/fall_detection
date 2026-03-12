#!/usr/bin/env python3
"""Export AlphaPose's YOLOv3-SPP detector to ONNX for Jetson / TensorRT.

Run this from the AlphaPose repository root so imports such as
`detector.yolo.darknet` resolve exactly as they do in the upstream repo.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export AlphaPose YOLOv3-SPP detector to ONNX."
    )
    parser.add_argument(
        "--cfg",
        required=True,
        help="Darknet cfg path, e.g. detector/yolo/cfg/yolov3-spp.cfg",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Darknet weights path, e.g. detector/yolo/data/yolov3-spp.weights",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output ONNX path, e.g. onnx/yolov3_spp.onnx",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version. Default: 13",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=608,
        help="Square detector input size. Default: 608",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Fixed export batch size. Default: 1",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help='Export device, e.g. "cuda:0" or "cpu". Default: auto-detect',
    )
    parser.add_argument(
        "--input-name",
        default="images",
        help="ONNX input tensor name. Default: images",
    )
    parser.add_argument(
        "--output-name",
        default="detections",
        help="ONNX output tensor name. Default: detections",
    )
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Run 'polygraphy surgeon sanitize --fold-constants' after export.",
    )
    parser.add_argument(
        "--sanitize-output",
        default="",
        help="Optional path for sanitized ONNX. Defaults to <output stem>_sanitized.onnx",
    )
    return parser.parse_args()


def ensure_file_exists(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def mkdir_for_file(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def format_onnx_dims(value_info) -> str:
    dims = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(str(dim.dim_value))
        else:
            dims.append("?")
    return "x".join(dims)


def print_onnx_io_summary(onnx_path: Path) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    print("\nONNX validation summary")
    print(f"  file: {onnx_path}")
    print("  inputs:")
    for value in model.graph.input:
        print(f"    - {value.name}: {format_onnx_dims(value)}")
    print("  outputs:")
    for value in model.graph.output:
        print(f"    - {value.name}: {format_onnx_dims(value)}")


class YoloExportWrapper(torch.nn.Module):
    """Bind the old AlphaPose Darknet forward signature to a single tensor input."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        super().__init__()
        self.model = model
        self.export_args = SimpleNamespace(device=device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images, self.export_args)


def build_detector(cfg_path: Path, weights_path: Path, input_size: int, device: torch.device) -> YoloExportWrapper:
    from detector.yolo.darknet import Darknet

    model = Darknet(str(cfg_path))
    model.load_weights(str(weights_path))
    model.net_info["height"] = str(input_size)
    model.net_info["width"] = str(input_size)
    model.eval()
    model.to(device)
    return YoloExportWrapper(model, device)


def maybe_sanitize(onnx_path: Path, sanitize_output: Path) -> Path:
    polygraphy = shutil.which("polygraphy")
    if polygraphy is None:
        raise RuntimeError(
            "--sanitize was requested but 'polygraphy' was not found in PATH. "
            "Install it first, for example: python3 -m pip install polygraphy"
        )

    cmd = [
        polygraphy,
        "surgeon",
        "sanitize",
        str(onnx_path),
        "--fold-constants",
        "--output",
        str(sanitize_output),
    ]
    print("\nRunning ONNX sanitize step")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return sanitize_output


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.input_size % 32 != 0:
        raise ValueError("--input-size should be a multiple of 32 for YOLOv3-SPP")

    cfg_path = ensure_file_exists(args.cfg, "Cfg")
    weights_path = ensure_file_exists(args.weights, "Weights")
    output_path = mkdir_for_file(args.output)
    sanitize_output = Path(args.sanitize_output) if args.sanitize_output else output_path.with_name(output_path.stem + "_sanitized.onnx")

    device = torch.device(args.device)
    print("YOLOv3-SPP ONNX export")
    print(f"  cfg:          {cfg_path}")
    print(f"  weights:      {weights_path}")
    print(f"  output:       {output_path}")
    print(f"  opset:        {args.opset}")
    print(f"  device:       {device}")
    print(f"  input shape:  {args.batch_size}x3x{args.input_size}x{args.input_size}")
    print(f"  sanitize:     {args.sanitize}")

    model = build_detector(cfg_path, weights_path, args.input_size, device)
    dummy = torch.randn(
        args.batch_size,
        3,
        args.input_size,
        args.input_size,
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        preview = model(dummy)
        print(f"  preview output tensor shape: {tuple(preview.shape)}")

        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=[args.input_name],
            output_names=[args.output_name],
        )

    print(f"\nExport complete: {output_path}")
    print_onnx_io_summary(output_path)

    if args.sanitize:
        sanitized = maybe_sanitize(output_path, sanitize_output)
        print(f"\nSanitized ONNX written to: {sanitized}")
        print_onnx_io_summary(sanitized)


if __name__ == "__main__":
    main()
