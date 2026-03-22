#!/usr/bin/env python3
"""
Export the two-stage Hugging Face ViTPose pipeline to ONNX.

Stage 1:
  RT-DETR person detector -> detector ONNX with input `pixel_values`
                             and outputs `logits`, `pred_boxes`

Stage 2:
  ViTPose pose estimator  -> pose ONNX with input `pixel_values`
                             and output `heatmaps`

The script writes:
  - ONNX files for both stages
  - a JSON manifest with paths and tensor metadata
  - a shell-friendly `.env` manifest consumed by the bash wrapper

Notes:
  - This exports model forward passes only. Hugging Face processor-side
    preprocessing and postprocessing are intentionally left outside the engines.
  - The RT-DETR export uses only `pixel_values`. The model creates an all-ones
    mask internally when `pixel_mask` is omitted, which matches the default
    fixed-size processor path used by this repo.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
import PIL.Image
import torch
from torch import nn

# Older Pillow builds on Jetson may not expose Image.Resampling, but recent
# transformers releases import it unconditionally.
if not hasattr(PIL.Image, "Resampling"):
    class _PillowResamplingCompat:
        NEAREST = PIL.Image.NEAREST
        BOX = getattr(PIL.Image, "BOX", PIL.Image.NEAREST)
        BILINEAR = PIL.Image.BILINEAR
        HAMMING = getattr(PIL.Image, "HAMMING", PIL.Image.BILINEAR)
        BICUBIC = PIL.Image.BICUBIC
        LANCZOS = getattr(PIL.Image, "LANCZOS", PIL.Image.BICUBIC)

    PIL.Image.Resampling = _PillowResamplingCompat

from transformers import RTDetrForObjectDetection, VitPoseForPoseEstimation


DEFAULT_DETECTOR_MODEL = "PekingU/rtdetr_r50vd_coco_o365"
DEFAULT_POSE_MODEL = "usyd-community/vitpose-base"
DEFAULT_DETECTOR_HW = (640, 640)
DEFAULT_POSE_HW = (256, 192)


@dataclass(frozen=True)
class StageExportInfo:
    stage_name: str
    model_id: str
    slug: str
    onnx_path: Path
    input_name: str
    output_names: list[str]
    channels: int
    height: int
    width: int


class RTDetrExportWrapper(nn.Module):
    def __init__(self, model: RTDetrForObjectDetection):
        super().__init__()
        self.model = model.eval()

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        return outputs.logits, outputs.pred_boxes


class VitPoseExportWrapper(nn.Module):
    def __init__(self, model: VitPoseForPoseEstimation):
        super().__init__()
        self.model = model.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values, return_dict=True)
        return outputs.heatmaps


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export RT-DETR and ViTPose stages to ONNX.")
    ap.add_argument("--detector-model", default=DEFAULT_DETECTOR_MODEL, help="Hugging Face RT-DETR model id.")
    ap.add_argument("--pose-model", default=DEFAULT_POSE_MODEL, help="Hugging Face ViTPose model id.")
    ap.add_argument(
        "--output-root",
        default=Path(__file__).resolve().parents[1] / "models" / "vitpose_trt",
        type=Path,
        help="Directory where ONNX files and manifests are written.",
    )
    ap.add_argument("--detector-height", type=int, default=None, help="Override detector input height.")
    ap.add_argument("--detector-width", type=int, default=None, help="Override detector input width.")
    ap.add_argument("--pose-height", type=int, default=None, help="Override pose input height.")
    ap.add_argument("--pose-width", type=int, default=None, help="Override pose input width.")
    ap.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used during ONNX export. CUDA is optional; CPU is the safe default.",
    )
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    ap.add_argument(
        "--static-batch",
        action="store_true",
        help="Export with a fixed batch dimension of 1 instead of a dynamic batch axis.",
    )
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Hugging Face models/processors from the local cache only.",
    )
    return ap.parse_args()


def slugify_model_id(model_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_").lower()
    return slug or "model"


def quote_shell(value: Any) -> str:
    return shlex.quote(str(value))


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for export, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_pose_hw_from_model(model: VitPoseForPoseEstimation, fallback: tuple[int, int]) -> tuple[int, int]:
    cfg = getattr(model, "config", None)
    backbone_cfg = getattr(cfg, "backbone_config", None)
    image_size = getattr(backbone_cfg, "image_size", None)
    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        height, width = image_size
        if isinstance(height, int) and isinstance(width, int):
            return int(height), int(width)
    return fallback


def export_onnx_model(
    model: nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: Path,
    input_name: str,
    output_names: list[str],
    dynamic_batch: bool,
    opset: int,
) -> None:
    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {input_name: {0: "batch"}}
        for output_name in output_names:
            dynamic_axes[output_name] = {0: "batch"}

    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        input_names=[input_name],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)


def write_manifest_files(
    output_root: Path,
    dynamic_batch: bool,
    detector_info: StageExportInfo,
    pose_info: StageExportInfo,
) -> tuple[Path, Path]:
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_json_path = manifest_dir / "vitpose_export_manifest.json"
    manifest_env_path = manifest_dir / "vitpose_export_manifest.env"

    manifest = {
        "dynamic_batch": dynamic_batch,
        "detector": {
            "stage_name": detector_info.stage_name,
            "model_id": detector_info.model_id,
            "slug": detector_info.slug,
            "onnx_path": str(detector_info.onnx_path),
            "input_name": detector_info.input_name,
            "output_names": detector_info.output_names,
            "channels": detector_info.channels,
            "height": detector_info.height,
            "width": detector_info.width,
        },
        "pose": {
            "stage_name": pose_info.stage_name,
            "model_id": pose_info.model_id,
            "slug": pose_info.slug,
            "onnx_path": str(pose_info.onnx_path),
            "input_name": pose_info.input_name,
            "output_names": pose_info.output_names,
            "channels": pose_info.channels,
            "height": pose_info.height,
            "width": pose_info.width,
        },
    }
    manifest_json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    env_lines = [
        f"EXPORT_ROOT={quote_shell(output_root)}",
        f"DYNAMIC_BATCH={1 if dynamic_batch else 0}",
        f"DETECTOR_STAGE_NAME={quote_shell(detector_info.stage_name)}",
        f"DETECTOR_MODEL_ID={quote_shell(detector_info.model_id)}",
        f"DETECTOR_SLUG={quote_shell(detector_info.slug)}",
        f"DETECTOR_ONNX={quote_shell(detector_info.onnx_path)}",
        f"DETECTOR_INPUT_NAME={quote_shell(detector_info.input_name)}",
        f"DETECTOR_CHANNELS={detector_info.channels}",
        f"DETECTOR_HEIGHT={detector_info.height}",
        f"DETECTOR_WIDTH={detector_info.width}",
        f"POSE_STAGE_NAME={quote_shell(pose_info.stage_name)}",
        f"POSE_MODEL_ID={quote_shell(pose_info.model_id)}",
        f"POSE_SLUG={quote_shell(pose_info.slug)}",
        f"POSE_ONNX={quote_shell(pose_info.onnx_path)}",
        f"POSE_INPUT_NAME={quote_shell(pose_info.input_name)}",
        f"POSE_CHANNELS={pose_info.channels}",
        f"POSE_HEIGHT={pose_info.height}",
        f"POSE_WIDTH={pose_info.width}",
    ]
    manifest_env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    return manifest_json_path, manifest_env_path


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    output_root = args.output_root.expanduser().resolve()
    onnx_dir = output_root / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    detector_model = RTDetrForObjectDetection.from_pretrained(
        args.detector_model,
        local_files_only=args.local_files_only,
    ).to(device).eval()

    pose_model = VitPoseForPoseEstimation.from_pretrained(
        args.pose_model,
        local_files_only=args.local_files_only,
    ).to(device).eval()

    detector_hw = DEFAULT_DETECTOR_HW
    if args.detector_height is not None and args.detector_width is not None:
        detector_hw = (args.detector_height, args.detector_width)
    elif args.detector_height is not None or args.detector_width is not None:
        raise ValueError("Override both --detector-height and --detector-width together.")

    pose_hw = resolve_pose_hw_from_model(pose_model, fallback=DEFAULT_POSE_HW)
    if args.pose_height is not None and args.pose_width is not None:
        pose_hw = (args.pose_height, args.pose_width)
    elif args.pose_height is not None or args.pose_width is not None:
        raise ValueError("Override both --pose-height and --pose-width together.")

    detector_slug = f"detector_{slugify_model_id(args.detector_model)}"
    pose_slug = f"pose_{slugify_model_id(args.pose_model)}"

    detector_onnx = (onnx_dir / f"{detector_slug}.onnx").resolve()
    pose_onnx = (onnx_dir / f"{pose_slug}.onnx").resolve()

    dynamic_batch = not args.static_batch
    export_batch = 1

    detector_dummy = torch.zeros(
        (export_batch, 3, detector_hw[0], detector_hw[1]),
        dtype=torch.float32,
        device=device,
    )
    pose_dummy = torch.zeros(
        (export_batch, 3, pose_hw[0], pose_hw[1]),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        export_onnx_model(
            RTDetrExportWrapper(detector_model),
            detector_dummy,
            detector_onnx,
            input_name="pixel_values",
            output_names=["logits", "pred_boxes"],
            dynamic_batch=dynamic_batch,
            opset=args.opset,
        )
        export_onnx_model(
            VitPoseExportWrapper(pose_model),
            pose_dummy,
            pose_onnx,
            input_name="pixel_values",
            output_names=["heatmaps"],
            dynamic_batch=dynamic_batch,
            opset=args.opset,
        )

    detector_info = StageExportInfo(
        stage_name="detector",
        model_id=args.detector_model,
        slug=detector_slug,
        onnx_path=detector_onnx,
        input_name="pixel_values",
        output_names=["logits", "pred_boxes"],
        channels=3,
        height=detector_hw[0],
        width=detector_hw[1],
    )
    pose_info = StageExportInfo(
        stage_name="pose",
        model_id=args.pose_model,
        slug=pose_slug,
        onnx_path=pose_onnx,
        input_name="pixel_values",
        output_names=["heatmaps"],
        channels=3,
        height=pose_hw[0],
        width=pose_hw[1],
    )

    manifest_json, manifest_env = write_manifest_files(
        output_root=output_root,
        dynamic_batch=dynamic_batch,
        detector_info=detector_info,
        pose_info=pose_info,
    )

    print(f"Exported detector ONNX: {detector_onnx}")
    print(f"Exported pose ONNX: {pose_onnx}")
    print(f"Wrote JSON manifest: {manifest_json}")
    print(f"Wrote env manifest: {manifest_env}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
