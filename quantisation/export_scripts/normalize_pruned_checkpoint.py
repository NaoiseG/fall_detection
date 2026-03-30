#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from importlib import metadata as importlib_metadata
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELOPT_ROOT = REPO_ROOT / "pruning" / "Model-Optimizer"
DEFAULT_ULTRALYTICS_ROOT = REPO_ROOT / "pruning" / "yolov11-prune"


def always_true() -> bool:
    """
    Compatibility shim for older pruned checkpoints that serialized an instance-level
    `is_fused = always_true` monkey patch from a script executed as `__main__`.
    """
    return True


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


def clear_modules(prefix: str) -> None:
    for module_name in list(sys.modules):
        if module_name == prefix or module_name.startswith(f"{prefix}."):
            sys.modules.pop(module_name, None)


def patch_modelopt_version_lookup() -> None:
    original_version = importlib_metadata.version

    def version_with_local_fallback(distribution_name: str) -> str:
        if distribution_name != "nvidia-modelopt":
            return original_version(distribution_name)
        try:
            return original_version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            return "0+local"

    importlib_metadata.version = version_with_local_fallback


def import_modelopt_modules(modelopt_root: str) -> tuple[Any, Any, str]:
    try:
        import modelopt.torch.nas as mtn
        import modelopt.torch.opt as mto

        return mtn, mto, "installed package"
    except Exception as first_exc:
        clear_modules("modelopt")
        prepend_sys_path(modelopt_root)
        patch_modelopt_version_lookup()

        try:
            import modelopt.torch.nas as mtn
            import modelopt.torch.opt as mto

            return mtn, mto, str(Path(modelopt_root).resolve())
        except Exception as second_exc:
            raise RuntimeError(
                "Could not import ModelOpt for checkpoint normalization. "
                f"Installed-package import failed with: {first_exc!r}. "
                f"Local checkout import failed with: {second_exc!r}."
            ) from second_exc


def clear_instance_override(model: Any, attr_name: str) -> None:
    if attr_name in getattr(model, "__dict__", {}):
        delattr(model, attr_name)


def get_last_modelopt_mode_name(model: Any) -> str | None:
    state = getattr(model, "_modelopt_state", None)
    if not state:
        return None
    try:
        return str(state[-1][0])
    except Exception:
        return None


def finalize_model(model: Any, *, nas_module: Any, opt_module: Any) -> Any:
    model = getattr(model, "module", model)
    clear_instance_override(model, "is_fused")

    last_mode_name = get_last_modelopt_mode_name(model)
    already_exported = last_mode_name in {"export", "export_nas"}

    if opt_module.ModeloptStateManager.is_converted(model) and not already_exported:
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


def save_normalized_checkpoint(checkpoint: dict[str, Any], output_path: Path, *, torch_module: Any) -> None:
    try:
        torch_module.save(checkpoint, output_path, use_dill=False)
    except TypeError as exc:
        if "use_dill" not in str(exc):
            raise
        torch_module.save(checkpoint, output_path)


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output checkpoint already exists: {output_path}")

    prepend_sys_path(args.ultralytics_root)

    import torch
    import ultralytics

    mtn, mto, modelopt_source = import_modelopt_modules(args.modelopt_root)

    print(f"[normalize] input:  {input_path}")
    print(f"[normalize] output: {output_path}")
    print(f"[normalize] modelopt: {modelopt_source}")
    print(f"[normalize] ultralytics: {Path(ultralytics.__file__).resolve()}")

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
    save_normalized_checkpoint(normalized_ckpt, output_path, torch_module=torch)

    print(f"[normalize] wrote:  {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
