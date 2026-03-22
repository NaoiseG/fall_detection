#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import PIL.Image
import torch
import yaml

try:
    import tensorrt as trt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERROR: TensorRT Python bindings are required (import tensorrt failed)") from exc

try:
    import pycuda.autoinit  # noqa: F401  # initializes CUDA context
    import pycuda.driver as cuda
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERROR: pycuda is required (import pycuda failed)") from exc

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

try:
    from transformers import AutoProcessor
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERROR: transformers is required (import AutoProcessor failed)") from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_DETECTOR_MODEL = "PekingU/rtdetr_r50vd_coco_o365"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a TensorRT INT8 calibration cache for RT-DETR from data.yaml.",
    )
    parser.add_argument("--onnx", required=True, help="Path to the RT-DETR ONNX file")
    parser.add_argument("--data", required=True, help="Path to calibration data.yaml")
    parser.add_argument("--cache", required=True, help="Output calibration cache path")
    parser.add_argument(
        "--detector-model",
        default=DEFAULT_DETECTOR_MODEL,
        help="Hugging Face RT-DETR model id used to create preprocessing pixel_values.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split from data.yaml to use. Default: val",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Calibration batch size. Default: 8")
    parser.add_argument(
        "--max-images",
        type=int,
        default=256,
        help="Maximum images to use for calibration. Default: 256",
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=2048,
        help="TensorRT workspace in MiB. Default: 2048",
    )
    parser.add_argument(
        "--input-name",
        default="",
        help="Optional ONNX input tensor name override. Default: first network input",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=0,
        help="Optional input height override. Default: infer from ONNX input shape",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=0,
        help="Optional input width override. Default: infer from ONNX input shape",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Hugging Face processor assets from the local cache only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing calibration cache",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose TensorRT logging",
    )
    return parser.parse_args()


def fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        fail(f"data.yaml is not a mapping: {path}")
    return data


def collect_images_from_path(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if path.is_file() and path.suffix.lower() == ".txt":
        items: List[Path] = []
        parent = path.parent
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path = Path(line)
                if not img_path.is_absolute():
                    img_path = (parent / img_path).resolve()
                items.append(img_path)
        return [p for p in items if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    return []


def collect_split_images(data_yaml: Path, split: str) -> List[Path]:
    data = read_yaml(data_yaml)
    if split not in data:
        fail(f"'{split}' not found in {data_yaml}")

    base_dir = data_yaml.parent.resolve()
    split_value = data[split]
    candidates: List[Path] = []

    if isinstance(split_value, str):
        candidates.extend(collect_images_from_path(resolve_path(base_dir, split_value)))
    elif isinstance(split_value, list):
        for item in split_value:
            if not isinstance(item, str):
                continue
            candidates.extend(collect_images_from_path(resolve_path(base_dir, item)))
    else:
        fail(f"Unsupported type for '{split}' in {data_yaml}: {type(split_value).__name__}")

    unique = sorted({p.resolve() for p in candidates if p.exists()})
    if not unique:
        fail(f"No images found for split '{split}' from {data_yaml}")
    return unique


def open_rgb_image(image_path: Path) -> PIL.Image.Image:
    try:
        with PIL.Image.open(image_path) as img:
            return img.convert("RGB")
    except Exception as exc:
        fail(f"Failed to read image {image_path}: {exc}")


class RTDetrBatchStream:
    def __init__(
        self,
        *,
        image_paths: Sequence[Path],
        batch_size: int,
        processor,
        input_height: int,
        input_width: int,
    ):
        self.image_paths = list(image_paths)
        self.batch_size = int(batch_size)
        self.processor = processor
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.max_batches = len(self.image_paths) // self.batch_size
        self.reset()

    def reset(self) -> None:
        self.batch_idx = 0

    def next_batch(self) -> np.ndarray | None:
        if self.batch_idx >= self.max_batches:
            return None
        start = self.batch_idx * self.batch_size
        end = start + self.batch_size
        batch_paths = self.image_paths[start:end]
        images = [open_rgb_image(path) for path in batch_paths]

        processor_kwargs = {}
        if self.input_height > 0 and self.input_width > 0:
            processor_kwargs["size"] = {"height": self.input_height, "width": self.input_width}

        encoded = self.processor(images=images, return_tensors="pt", **processor_kwargs)
        pixel_values = encoded.get("pixel_values")
        if pixel_values is None:
            fail("Processor output does not contain 'pixel_values'")
        batch = np.ascontiguousarray(pixel_values.detach().cpu().numpy(), dtype=np.float32)
        self.batch_idx += 1
        return batch


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batchstream: RTDetrBatchStream, cache_file: Path, input_name: str):
        super().__init__()
        self.batchstream = batchstream
        self.cache_file = cache_file
        self.input_name = input_name
        self.device_input = cuda.mem_alloc(
            batchstream.batch_size
            * 3
            * batchstream.input_height
            * batchstream.input_width
            * np.dtype(np.float32).itemsize
        )

    def get_batch_size(self) -> int:
        return self.batchstream.batch_size

    def get_batch(self, names: Sequence[str]) -> List[int] | None:
        batch = self.batchstream.next_batch()
        if batch is None:
            return None
        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if self.cache_file.is_file():
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(cache)


def get_network_input(network, preferred_name: str) -> tuple[str, int, int, int]:
    if network.num_inputs < 1:
        fail("Parsed network has no inputs")

    selected = None
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        if preferred_name and tensor.name == preferred_name:
            selected = tensor
            break
        if selected is None:
            selected = tensor

    if selected is None:
        fail("Could not resolve network input tensor")

    shape = tuple(int(v) for v in selected.shape)
    if len(shape) != 4:
        fail(f"Expected 4D network input, got {shape} for tensor '{selected.name}'")
    batch_dim, channels, height, width = shape
    if channels != 3:
        fail(f"Expected 3-channel input, got shape {shape} for tensor '{selected.name}'")
    return selected.name, int(batch_dim), int(height), int(width)


def set_workspace(config, workspace_mb: int) -> None:
    workspace_bytes = int(workspace_mb) * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:  # TensorRT 8.x fallback
        config.max_workspace_size = workspace_bytes


def build_cache(
    *,
    onnx_path: Path,
    cache_path: Path,
    image_paths: Sequence[Path],
    detector_model: str,
    batch_size: int,
    workspace_mb: int,
    input_name_override: str,
    input_height_override: int,
    input_width_override: int,
    local_files_only: bool,
    verbose: bool,
) -> None:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    if not onnx_path.is_file():
        fail(f"ONNX not found: {onnx_path}")

    print("Parsing ONNX")
    print(f"  file: {onnx_path}")
    ok = parser.parse(onnx_path.read_bytes())
    if not ok:
        print("TensorRT ONNX parse errors:", file=sys.stderr)
        for i in range(parser.num_errors):
            print(f"  - {parser.get_error(i)}", file=sys.stderr)
        fail("Failed to parse ONNX")

    input_name, input_batch_dim, inferred_height, inferred_width = get_network_input(network, input_name_override)
    input_height = input_height_override or inferred_height
    input_width = input_width_override or inferred_width

    if input_height < 1 or input_width < 1:
        fail(
            "Dynamic or invalid spatial input shape was detected. "
            "Pass both --input-height and --input-width or export fixed-shape ONNX."
        )

    effective_batch_size = int(batch_size)
    if input_batch_dim > 0:
        effective_batch_size = int(input_batch_dim)
    if effective_batch_size < 1:
        fail("Resolved calibration batch size must be >= 1")

    usable_count = min(len(image_paths), max(effective_batch_size, (len(image_paths) // effective_batch_size) * effective_batch_size))
    if usable_count < effective_batch_size:
        fail(
            f"Need at least one full calibration batch: found {len(image_paths)} images, "
            f"batch-size is {effective_batch_size}"
        )
    image_paths = list(image_paths[:usable_count])

    print("Calibration settings")
    print(f"  detector model: {detector_model}")
    print(f"  input tensor:   {input_name}")
    print(f"  input shape:    {effective_batch_size}x3x{input_height}x{input_width}")
    print(f"  images used:    {len(image_paths)}")
    print(f"  batches used:   {len(image_paths) // effective_batch_size}")
    print(f"  workspace MiB:  {workspace_mb}")
    print(f"  cache output:   {cache_path}")

    processor = AutoProcessor.from_pretrained(detector_model, local_files_only=local_files_only)
    stream = RTDetrBatchStream(
        image_paths=image_paths,
        batch_size=effective_batch_size,
        processor=processor,
        input_height=input_height,
        input_width=input_width,
    )
    calibrator = EntropyCalibrator(stream, cache_path, input_name)

    config = builder.create_builder_config()
    set_workspace(config, workspace_mb)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    if input_batch_dim < 0 or inferred_height < 0 or inferred_width < 0:
        profile = builder.create_optimization_profile()
        min_shape = (1, 3, input_height, input_width)
        opt_shape = (effective_batch_size, 3, input_height, input_width)
        max_shape = (effective_batch_size, 3, input_height, input_width)
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    print("Building temporary TensorRT engine to generate calibration cache...")
    engine_blob = None
    if hasattr(builder, "build_serialized_network"):
        engine_blob = builder.build_serialized_network(network, config)
    else:  # older TRT fallback
        engine = builder.build_engine(network, config)
        if engine is not None:
            engine_blob = engine.serialize()

    if engine_blob is None:
        fail("TensorRT engine build failed; calibration cache was not generated")
    if not cache_path.is_file():
        fail("Engine build finished but calibration cache file was not written")


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        fail("--batch-size must be >= 1")
    if args.max_images < 1:
        fail("--max-images must be >= 1")
    if args.input_height < 0 or args.input_width < 0:
        fail("--input-height and --input-width must be >= 0")
    if bool(args.input_height) != bool(args.input_width):
        fail("Set both --input-height and --input-width together.")

    onnx_path = Path(args.onnx).resolve()
    data_yaml = Path(args.data).resolve()
    cache_path = Path(args.cache).resolve()

    if cache_path.exists() and not args.force:
        print(f"Calibration cache already exists: {cache_path}")
        print("Use --force to regenerate it.")
        return

    if not data_yaml.is_file():
        fail(f"data.yaml not found: {data_yaml}")

    all_images = collect_split_images(data_yaml, args.split)
    selected_images = all_images[: args.max_images]
    print("Calibration dataset")
    print(f"  data yaml:      {data_yaml}")
    print(f"  split:          {args.split}")
    print(f"  images found:   {len(all_images)}")
    print(f"  images chosen:  {len(selected_images)}")
    if selected_images:
        print(f"  first image:    {selected_images[0]}")

    build_cache(
        onnx_path=onnx_path,
        cache_path=cache_path,
        image_paths=selected_images,
        detector_model=args.detector_model,
        batch_size=args.batch_size,
        workspace_mb=args.workspace_mb,
        input_name_override=args.input_name,
        input_height_override=args.input_height,
        input_width_override=args.input_width,
        local_files_only=bool(args.local_files_only),
        verbose=bool(args.verbose),
    )

    print("\nDone.")
    print(f"Calibration cache written to: {cache_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
