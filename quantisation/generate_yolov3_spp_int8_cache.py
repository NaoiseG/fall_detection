#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import yaml

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERROR: opencv-python is required (import cv2 failed)") from exc

try:
    import tensorrt as trt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERROR: TensorRT Python bindings are required (import tensorrt failed)") from exc

try:
    import pycuda.autoinit  # noqa: F401  # initializes CUDA context
    import pycuda.driver as cuda
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERROR: pycuda is required (import pycuda failed)") from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a TensorRT INT8 calibration cache for AlphaPose YOLOv3-SPP from data.yaml.",
    )
    parser.add_argument("--onnx", required=True, help="Path to yolov3_spp.onnx")
    parser.add_argument("--data", required=True, help="Path to calibration data.yaml")
    parser.add_argument(
        "--cache",
        required=True,
        help="Output calibration cache path, e.g. calibration_dataset_upfall/yolov3_spp_int8.cache",
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
        "--input-size",
        type=int,
        default=0,
        help="Optional square input size override. Default: infer from ONNX input shape",
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


def letterbox_image(img: np.ndarray, size: int) -> np.ndarray:
    img_h, img_w = img.shape[:2]
    scale = min(size / img_w, size / img_h)
    new_w = max(int(img_w * scale), 1)
    new_h = max(int(img_h * scale), 1)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w, :] = resized
    return canvas


def preprocess_image(image_path: Path, input_size: int) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        fail(f"Failed to read image: {image_path}")
    # Match AlphaPose detector/yolo/preprocess.py: letterbox, BGR->RGB, CHW, /255.
    img = letterbox_image(img, input_size)
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = img.astype(np.float32) / 255.0
    return img


class ImageBatchStream:
    def __init__(self, image_paths: Sequence[Path], batch_size: int, input_size: int):
        self.image_paths = list(image_paths)
        self.batch_size = batch_size
        self.input_size = input_size
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
        batch = np.ascontiguousarray(
            np.stack([preprocess_image(p, self.input_size) for p in batch_paths], axis=0),
            dtype=np.float32,
        )
        self.batch_idx += 1
        return batch


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batchstream: ImageBatchStream, cache_file: Path, input_name: str):
        super().__init__()
        self.batchstream = batchstream
        self.cache_file = cache_file
        self.input_name = input_name
        self.device_input = cuda.mem_alloc(
            batchstream.batch_size * 3 * batchstream.input_size * batchstream.input_size * np.dtype(np.float32).itemsize
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



def get_network_input(network, preferred_name: str) -> tuple[str, int]:
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

    shape = tuple(selected.shape)
    if len(shape) != 4:
        fail(f"Expected 4D network input, got {shape} for tensor '{selected.name}'")
    _, c, h, w = shape
    if c != 3:
        fail(f"Expected 3-channel input, got shape {shape} for tensor '{selected.name}'")
    if h != w:
        fail(f"Expected square input for YOLOv3-SPP, got shape {shape} for tensor '{selected.name}'")
    if h < 1:
        fail(f"Dynamic or invalid spatial shape {shape}; pass --input-size explicitly or export fixed-shape ONNX")
    return selected.name, int(h)



def set_workspace(config, workspace_mb: int) -> None:
    workspace_bytes = int(workspace_mb) * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:  # TensorRT 8.x fallback
        config.max_workspace_size = workspace_bytes



def build_cache(
    onnx_path: Path,
    cache_path: Path,
    image_paths: Sequence[Path],
    batch_size: int,
    workspace_mb: int,
    input_name_override: str,
    input_size_override: int,
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

    input_name, inferred_size = get_network_input(network, input_name_override)
    input_size = input_size_override or inferred_size

    usable_count = min(len(image_paths), max(batch_size, (len(image_paths) // batch_size) * batch_size))
    if usable_count < batch_size:
        fail(
            f"Need at least one full calibration batch: found {len(image_paths)} images, batch-size is {batch_size}"
        )
    image_paths = list(image_paths[:usable_count])

    print("Calibration settings")
    print(f"  input tensor:   {input_name}")
    print(f"  input size:     {input_size}x{input_size}")
    print(f"  batch size:     {batch_size}")
    print(f"  images used:    {len(image_paths)}")
    print(f"  batches used:   {len(image_paths) // batch_size}")
    print(f"  workspace MiB:  {workspace_mb}")
    print(f"  cache output:   {cache_path}")

    stream = ImageBatchStream(image_paths=image_paths, batch_size=batch_size, input_size=input_size)
    calibrator = EntropyCalibrator(stream, cache_path, input_name)

    config = builder.create_builder_config()
    set_workspace(config, workspace_mb)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

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
        batch_size=args.batch_size,
        workspace_mb=args.workspace_mb,
        input_name_override=args.input_name,
        input_size_override=args.input_size,
        verbose=args.verbose,
    )

    print("\nDone.")
    print(f"Calibration cache written to: {cache_path}")


if __name__ == "__main__":
    main()
