#!/usr/bin/env python3
"""Export AlphaPose FastPose to ONNX for Jetson / TensorRT.

Run this from the AlphaPose repository root so imports such as
`alphapose.models` and `alphapose.utils.config` resolve exactly the same way
as in the upstream repo.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export AlphaPose FastPose to ONNX."
    )
    parser.add_argument(
        "--cfg",
        required=True,
        help="AlphaPose pose config, e.g. configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="FastPose checkpoint, e.g. pretrained_models/fast_res50_256x192.pth",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output ONNX path, e.g. onnx/fastpose.onnx",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version. Default: 13",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Input height. Default: 256",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=192,
        help="Input width. Default: 192",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy export batch size for fixed-batch export. Default: 1",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export ONNX with a dynamic batch axis for person crops.",
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="Suggested min batch for TensorRT profile when --dynamic-batch is used.",
    )
    parser.add_argument(
        "--opt-batch",
        type=int,
        default=8,
        help="Suggested opt batch for TensorRT profile when --dynamic-batch is used.",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=32,
        help="Suggested max batch for TensorRT profile when --dynamic-batch is used.",
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
        default="heatmaps",
        help="ONNX output tensor name. Default: heatmaps",
    )
    return parser.parse_args()


@contextlib.contextmanager
def disable_torchvision_pretrained_downloads():
    """Patch torchvision ResNet constructors so AlphaPose does not try to download
    ImageNet weights during model construction.

    FastPose's constructor instantiates a torchvision ResNet with
    `pretrained=True`. For export we immediately load the actual AlphaPose
    checkpoint afterwards, so random initialization is fine and avoids any
    network dependency.
    """

    import torchvision.models as tv_models

    patched: Dict[str, Any] = {}
    resnet_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

    def make_wrapper(fn):
        signature = inspect.signature(fn)

        def wrapper(*args, **kwargs):
            kwargs.pop("pretrained", None)
            if "weights" in signature.parameters and "weights" not in kwargs:
                kwargs["weights"] = None
            return fn(*args, **kwargs)

        return wrapper

    try:
        for name in resnet_names:
            if hasattr(tv_models, name):
                original = getattr(tv_models, name)
                patched[name] = original
                setattr(tv_models, name, make_wrapper(original))
        yield
    finally:
        for name, original in patched.items():
            setattr(tv_models, name, original)


def ensure_file_exists(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def mkdir_for_file(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint_state_dict(checkpoint_path: Path, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Unsupported checkpoint format in {checkpoint_path}. Expected a state_dict or a dict containing one."
        )

    cleaned = {}
    for key, value in checkpoint.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


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


def build_pose_model(cfg_path: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    from alphapose.models import builder
    from alphapose.utils.config import update_config

    cfg = update_config(str(cfg_path))

    with disable_torchvision_pretrained_downloads():
        model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    state_dict = load_checkpoint_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys while loading FastPose checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading FastPose checkpoint: {unexpected}")

    model.eval()
    model.to(device)
    return model


def main() -> None:
    args = parse_args()

    cfg_path = ensure_file_exists(args.cfg, "Config")
    checkpoint_path = ensure_file_exists(args.checkpoint, "Checkpoint")
    output_path = mkdir_for_file(args.output)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if min(args.min_batch, args.opt_batch, args.max_batch) < 1:
        raise ValueError("--min-batch, --opt-batch and --max-batch must all be >= 1")
    if not (args.min_batch <= args.opt_batch <= args.max_batch):
        raise ValueError("Expected min_batch <= opt_batch <= max_batch")

    device = torch.device(args.device)
    print("FastPose ONNX export")
    print(f"  cfg:          {cfg_path}")
    print(f"  checkpoint:   {checkpoint_path}")
    print(f"  output:       {output_path}")
    print(f"  opset:        {args.opset}")
    print(f"  device:       {device}")
    print(f"  input shape:  {args.batch_size if not args.dynamic_batch else 1}x3x{args.height}x{args.width}")
    print(f"  dynamic:      {args.dynamic_batch}")

    model = build_pose_model(cfg_path, checkpoint_path, device)

    export_batch = 1 if args.dynamic_batch else args.batch_size
    dummy = torch.randn(export_batch, 3, args.height, args.width, device=device, dtype=torch.float32)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            args.input_name: {0: "batch"},
            args.output_name: {0: "batch"},
        }

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
            dynamic_axes=dynamic_axes,
        )

    print(f"\nExport complete: {output_path}")
    print_onnx_io_summary(output_path)

    if args.dynamic_batch:
        print("\nSuggested trtexec profile for FastPose")
        print(
            "  "
            f"--minShapes={args.input_name}:{args.min_batch}x3x{args.height}x{args.width} "
            f"--optShapes={args.input_name}:{args.opt_batch}x3x{args.height}x{args.width} "
            f"--maxShapes={args.input_name}:{args.max_batch}x3x{args.height}x{args.width} "
            f"--shapes={args.input_name}:{args.opt_batch}x3x{args.height}x{args.width}"
        )


if __name__ == "__main__":
    main()
