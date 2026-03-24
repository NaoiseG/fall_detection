#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELOPT_ROOT = REPO_ROOT / "pruning" / "Model-Optimizer"
DEFAULT_ULTRALYTICS_ROOT = REPO_ROOT / "pruning" / "yolov11-prune"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a ModelOpt-pruned YOLO checkpoint into a plain Ultralytics-style "
            "checkpoint that can be exported without ModelOpt runtime dependencies."
        )
    )
    parser.add_argument("--input", required=True, help="Input checkpoint path")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument(
        "--modelopt-root",
        default=str(DEFAULT_MODELOPT_ROOT),
        help="Path to the local Model-Optimizer checkout",
    )
    parser.add_argument(
        "--ultralytics-root",
        default=str(DEFAULT_ULTRALYTICS_ROOT),
        help="Path to the pruning-time Ultralytics checkout",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output if it exists")
    return parser.parse_args()


def prepend_sys_path(path_text: str) -> None:
    if not path_text:
        return

    path = Path(path_text).resolve()
    if not path.exists():
        return

    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def clear_instance_override(model: Any, attr_name: str) -> None:
    if attr_name in getattr(model, "__dict__", {}):
        delattr(model, attr_name)


def finalize_model(model: Any, *, nas_module: Any, opt_module: Any) -> Any:
    model = getattr(model, "module", model)
    clear_instance_override(model, "is_fused")

    if opt_module.ModeloptStateManager.is_converted(model):
        model = nas_module.export(model)

    if hasattr(model, "_modelopt_state"):
        opt_module.ModeloptStateManager.remove_state(model)
    if hasattr(model, "_modelopt_state_version"):
        delattr(model, "_modelopt_state_version")

    clear_instance_override(model, "is_fused")
    if hasattr(model, "criterion"):
        model.criterion = None
    return model


def build_normalized_checkpoint(ckpt: dict[str, Any], model: Any) -> dict[str, Any]:
    model_to_save = copy.deepcopy(model).cpu().eval().half()
    clear_instance_override(model_to_save, "is_fused")
    if hasattr(model_to_save, "criterion"):
        model_to_save.criterion = None
    for param in model_to_save.parameters():
        param.requires_grad = False

    normalized_ckpt = dict(ckpt)
    normalized_ckpt["model"] = None
    normalized_ckpt["ema"] = model_to_save
    normalized_ckpt["optimizer"] = None
    normalized_ckpt["updates"] = ckpt.get("updates", 0)
    return normalized_ckpt


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output checkpoint already exists: {output_path}")

    prepend_sys_path(args.ultralytics_root)
    prepend_sys_path(args.modelopt_root)

    import torch
    import modelopt.torch.nas as mtn
    import modelopt.torch.opt as mto

    print(f"[normalize] input:  {input_path}")
    print(f"[normalize] output: {output_path}")

    try:
        ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(input_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected a checkpoint dict, got {type(ckpt).__name__}")

    model = ckpt.get("ema") or ckpt.get("model")
    if model is None:
        raise RuntimeError("Checkpoint does not contain an `ema` or `model` entry")

    model = finalize_model(model, nas_module=mtn, opt_module=mto)
    normalized_ckpt = build_normalized_checkpoint(ckpt, model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(normalized_ckpt, output_path, use_dill=False)

    print(f"[normalize] wrote:  {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
