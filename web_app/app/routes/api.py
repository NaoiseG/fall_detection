from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context
from werkzeug.utils import secure_filename

from app.config import (
    CLASSIFICATION_MODELS,
    CLASSIFICATION_WEIGHT_FILENAMES,
    DEFAULT_KEYPOINT_PRECISION,
    KEYPOINT_MODELS,
    KEYPOINT_PRECISIONS,
    KEYPOINT_TO_CLASSIFICATION_SUBDIR,
    KEYPOINT_YOLO_FP16_WEIGHT_FILENAMES,
    KEYPOINT_YOLO_WEIGHT_FILENAMES,
    VIDEO_EXTENSIONS,
    get_test_videos_dir,
    get_web_app_root,
)
from app.services.inference_service import InferenceService
from app.services.inference_stream_jobs import InferenceStreamJobManager
from app.services.preprocessing import preprocess_file, preprocess_json


api_bp = Blueprint("api", __name__, url_prefix="/api")
inference_service = InferenceService()
inference_service.load()
inference_stream_job_manager = InferenceStreamJobManager()


def _json_error(message: str, status_code: int):
    return jsonify({"ok": False, "error": message}), status_code


def _list_available_test_videos() -> Tuple[Path, list[str]]:
    test_videos_dir = get_test_videos_dir()
    if not test_videos_dir.exists() or not test_videos_dir.is_dir():
        raise FileNotFoundError(f"Test video directory does not exist: {test_videos_dir}")

    videos = sorted(
        file_path.name
        for file_path in test_videos_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return test_videos_dir, videos


def _is_safe_video_filename(video_name: Any) -> bool:
    if not isinstance(video_name, str):
        return False

    candidate = video_name.strip()
    if not candidate:
        return False
    if "/" in candidate or "\\" in candidate or ".." in candidate:
        return False
    return Path(candidate).name == candidate


def _classification_models_root() -> Path:
    return (get_web_app_root() / "models" / "classification").resolve()


def _keypoint_models_root() -> Path:
    return (get_web_app_root() / "models" / "keypoint").resolve()


def _infer_keypoint_backend(model_name: str, model_path: Path) -> str:
    model_key = str(model_name).strip().lower()
    if "alphapose" in model_key:
        return "alphapose"
    if "vitpose" in model_key:
        return "vitpose"

    path = Path(model_path).expanduser()
    path_str = str(path).lower()
    if path.is_dir():
        if (path / "alphapose").is_dir() or path.name.lower() == "alphapose":
            return "alphapose"
        if "vitpose" in path.name.lower():
            return "vitpose"
    if "alphapose" in path_str:
        return "alphapose"
    if "vitpose" in path_str:
        return "vitpose"
    return "yolo"


def _resolve_relative_path(root: Path, relative_path: str, model_kind: str) -> Path:
    rel_path = Path(relative_path)
    if rel_path.is_absolute():
        raise ValueError(f"{model_kind} path must be relative.")

    resolved = (root / rel_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Invalid {model_kind} model path.") from error

    return resolved


def _find_named_file_recursive(*, search_root: Path, expected_filename: str, model_kind: str) -> Path:
    if not search_root.exists() or not search_root.is_dir():
        raise FileNotFoundError(f"Missing {model_kind} directory: {search_root}")

    matches = [path for path in search_root.rglob(expected_filename) if path.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"Missing {model_kind} weights file '{expected_filename}' under: {search_root}"
        )

    # Deterministic preference: shallowest match first, then lexical path order.
    matches.sort(key=lambda path: (len(path.relative_to(search_root).parts), path.as_posix().lower()))
    return matches[0]


def _resolve_classification_weights_path(*, classification_model: str, keypoint_model: str) -> Path:
    classification_root = _classification_models_root()
    model_dir_rel = CLASSIFICATION_MODELS[classification_model]
    model_dir = _resolve_relative_path(
        classification_root,
        model_dir_rel,
        "classification",
    )
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(
            f"Missing classification model directory for '{classification_model}': {model_dir}"
        )

    keypoint_variant = KEYPOINT_TO_CLASSIFICATION_SUBDIR.get(keypoint_model)
    if not keypoint_variant:
        raise ValueError(
            f"Missing classification subdirectory mapping for keypoint model: {keypoint_model}"
        )

    variant_dir = (model_dir / keypoint_variant).resolve()
    try:
        variant_dir.relative_to(model_dir)
    except ValueError as error:
        raise ValueError("Invalid classification variant directory.") from error

    if not variant_dir.exists() or not variant_dir.is_dir():
        raise FileNotFoundError(
            "Missing classification weights directory for "
            f"classification_model='{classification_model}' and "
            f"keypoint_model='{keypoint_model}': {variant_dir}"
        )

    expected_filename = CLASSIFICATION_WEIGHT_FILENAMES.get(classification_model)
    if not expected_filename:
        raise ValueError(f"Missing expected classification filename mapping for: {classification_model}")

    return _find_named_file_recursive(
        search_root=variant_dir,
        expected_filename=expected_filename,
        model_kind=f"classification model '{classification_model}' ({keypoint_variant})",
    )


def _validate_alphapose_bundle(root: Path) -> None:
    required = (
        "alphapose/__init__.py",
        "configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml",
        "pretrained_models/fast_res50_256x192.pth",
        "detector/yolo/cfg/yolov3-spp.cfg",
        "detector/yolo/data/yolov3-spp.weights",
    )
    missing = [item for item in required if not (root / item).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Incomplete AlphaPose bundle under {root}. Missing: {missing_str}"
        )


def _normalize_keypoint_precision(raw_precision: Any) -> str:
    if raw_precision is None:
        return str(DEFAULT_KEYPOINT_PRECISION)
    if not isinstance(raw_precision, str):
        raise ValueError("Invalid keypoint precision selection.")

    precision = str(raw_precision).strip().upper()
    if precision not in set(KEYPOINT_PRECISIONS):
        raise ValueError("Invalid keypoint precision selection.")
    return precision


def _resolve_yolo_weight_filename(*, model_name: str, keypoint_precision: str) -> str:
    precision = _normalize_keypoint_precision(keypoint_precision)
    if precision == "FP16":
        fp16_filename = KEYPOINT_YOLO_FP16_WEIGHT_FILENAMES.get(model_name)
        if fp16_filename:
            return fp16_filename
        raise ValueError(
            f"FP16 precision is not available for keypoint model: {model_name}"
        )

    expected_filename = KEYPOINT_YOLO_WEIGHT_FILENAMES.get(model_name)
    if expected_filename:
        return expected_filename
    raise ValueError(f"Missing expected YOLO filename mapping for keypoint model: {model_name}")


def _ensure_keypoint_asset(*, model_name: str, relative_path: str, keypoint_precision: str) -> Path:
    keypoint_root = _keypoint_models_root()
    weights_path = _resolve_relative_path(keypoint_root, relative_path, "keypoint")

    backend = _infer_keypoint_backend(model_name, weights_path)

    if backend == "vitpose":
        if _normalize_keypoint_precision(keypoint_precision) != "FP32":
            raise ValueError("FP16 precision is only supported for Ultralytics YOLO keypoint models.")
        if weights_path.exists() and weights_path.is_file():
            raise ValueError("Invalid ViTPose model path. Expected a directory.")
        weights_path.mkdir(parents=True, exist_ok=True)
        return weights_path

    if backend == "alphapose":
        if _normalize_keypoint_precision(keypoint_precision) != "FP32":
            raise ValueError("FP16 precision is only supported for Ultralytics YOLO keypoint models.")
        if not weights_path.exists() or not weights_path.is_dir():
            raise FileNotFoundError(f"Missing AlphaPose bundle directory: {weights_path}")
        _validate_alphapose_bundle(weights_path)
        return weights_path

    expected_filename = _resolve_yolo_weight_filename(
        model_name=model_name,
        keypoint_precision=keypoint_precision,
    )

    # Backward compatibility: accept explicit file paths if they match the expected filename.
    if weights_path.exists() and weights_path.is_file():
        if weights_path.name.lower() != expected_filename.lower():
            raise FileNotFoundError(
                f"Unexpected keypoint checkpoint filename for '{model_name}': {weights_path.name}. "
                f"Expected '{expected_filename}'."
            )
        return weights_path

    if not weights_path.exists() or not weights_path.is_dir():
        raise FileNotFoundError(
            f"Missing Ultralytics keypoint directory for '{model_name}': {weights_path}"
        )

    return _find_named_file_recursive(
        search_root=weights_path,
        expected_filename=expected_filename,
        model_kind=f"keypoint model '{model_name}'",
    )


def _validate_float(value: Any, *, name: str, min_value: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name}.") from error

    if not math.isfinite(out) or out < float(min_value):
        raise ValueError(f"Invalid {name}.")
    return out


def _validate_int(value: Any, *, name: str, min_value: int = 0) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name}.") from error

    if out < int(min_value):
        raise ValueError(f"Invalid {name}.")
    return out


def _prepare_stream_request(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid JSON body. Expected an object.")

    classification_model = payload.get("classification_model")
    keypoint_model = payload.get("keypoint_model")
    keypoint_precision = _normalize_keypoint_precision(
        payload.get("keypoint_precision", payload.get("precision", DEFAULT_KEYPOINT_PRECISION))
    )
    video_name = payload.get("video")
    if video_name is None:
        video_name = payload.get("video_name")

    if classification_model not in CLASSIFICATION_MODELS:
        raise ValueError("Invalid classification model selection.")
    if keypoint_model not in KEYPOINT_MODELS:
        raise ValueError("Invalid keypoint model selection.")
    if not _is_safe_video_filename(video_name):
        raise ValueError("Invalid video value. Expected a filename without path separators.")

    selected_video = str(video_name).strip()
    test_videos_dir, available_videos = _list_available_test_videos()
    if selected_video not in available_videos:
        raise ValueError("Selected video is not available in the test video directory.")

    resolved_video_path = (test_videos_dir / selected_video).resolve()
    try:
        resolved_video_path.relative_to(test_videos_dir)
    except ValueError as error:
        raise ValueError("Invalid video path resolution.") from error
    if not resolved_video_path.is_file():
        raise ValueError("Selected video file does not exist.")

    keypoint_weights_path = _ensure_keypoint_asset(
        model_name=keypoint_model,
        relative_path=KEYPOINT_MODELS[keypoint_model],
        keypoint_precision=keypoint_precision,
    )
    classification_weights_path = _resolve_classification_weights_path(
        classification_model=classification_model,
        keypoint_model=keypoint_model,
    )
    keypoint_backend = _infer_keypoint_backend(keypoint_model, keypoint_weights_path)

    realtime = bool(payload.get("realtime", True))
    display_fps = _validate_float(payload.get("display_fps", 0.0), name="display_fps", min_value=0.0)
    window_size = _validate_int(payload.get("T", 64), name="T", min_value=1)
    sampling_k = _validate_int(payload.get("k", 1), name="k", min_value=1)
    overlap_percent = _validate_float(
        payload.get("overlap_percent", payload.get("stride", 50.0)),
        name="stride",
        min_value=0.0,
    )
    if overlap_percent >= 100.0:
        raise ValueError("Invalid stride. Overlap percent must be less than 100.")
    stride_frames = max(1, int(round(float(window_size) * (1.0 - (float(overlap_percent) / 100.0)))))
    stride_frames = min(int(window_size), int(stride_frames))

    inference_options = {
        "display_fps": float(display_fps),
        "realtime": bool(realtime),
        "keypoint_backend": keypoint_backend,
        "max_people": 1,
        "T": int(window_size),
        "stride": int(stride_frames),
        "frame_step": int(sampling_k),
        "normalize_mode": "paper_rp",
        "rp_center_mode": "pixel",
        "rp_img_w": 640,
        "rp_img_h": 480,
        "missing_mode": "zeros_only",
        "interp_mode": "paper_group_linear",
        "interp_group": 100,
    }

    return {
        "video_name": selected_video,
        "video_path": resolved_video_path,
        "classification_model": classification_model,
        "keypoint_model": keypoint_model,
        "keypoint_precision": keypoint_precision,
        "classification_model_path": classification_weights_path,
        "keypoint_model_path": keypoint_weights_path,
        "inference_options": inference_options,
    }


def _start_stream_job_from_payload(payload: Any):
    prepared = _prepare_stream_request(payload)
    job = inference_stream_job_manager.start_job(
        video_path=prepared["video_path"],
        classification_model=prepared["classification_model"],
        classification_model_path=prepared["classification_model_path"],
        keypoint_model_path=prepared["keypoint_model_path"],
        inference_options=prepared["inference_options"],
    )
    return (
        jsonify(
            {
                "ok": True,
                "job_id": job.job_id,
                "stream_url": f"/api/stream/{job.job_id}",
                "status_url": f"/api/job_status/{job.job_id}",
            }
        ),
        202,
    )


@api_bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@api_bp.get("/list_test_videos")
def list_test_videos():
    try:
        _, videos = _list_available_test_videos()
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500
    except OSError as error:
        return jsonify({"error": f"Failed to list test videos: {error}"}), 500

    return jsonify({"videos": videos})


@api_bp.post("/start_inference_stream")
def start_inference_stream():
    try:
        payload = request.get_json(silent=True)
        return _start_stream_job_from_payload(payload)
    except ValueError as error:
        return _json_error(str(error), 400)
    except FileNotFoundError as error:
        return _json_error(str(error), 500)
    except OSError as error:
        return _json_error(f"Failed to start inference stream: {error}", 500)
    except Exception as error:
        return _json_error(f"Unexpected inference error: {error}", 500)


@api_bp.post("/run_inference")
def run_inference():
    # Backward-compatible alias.
    return start_inference_stream()


@api_bp.get("/stream/<job_id>")
def stream_inference(job_id: str):
    generator = inference_stream_job_manager.stream_generator(job_id)
    if generator is None:
        return jsonify({"error": "Job not found."}), 404

    response = Response(stream_with_context(generator), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response


@api_bp.get("/job_status/<job_id>")
def job_status(job_id: str):
    status = inference_stream_job_manager.get_status(job_id)
    if status is None:
        return jsonify({"error": "Job not found."}), 404
    return jsonify(status)


@api_bp.post("/predict")
def predict():
    uploaded_file = request.files.get("file")
    if uploaded_file and uploaded_file.filename:
        filename = secure_filename(uploaded_file.filename)
        if not filename:
            return jsonify({"error": "invalid filename"}), 400

        upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / filename
        uploaded_file.save(file_path)
        file_size = file_path.stat().st_size

        processed_path = preprocess_file(file_path)
        result = inference_service.predict_from_file(processed_path)

        return jsonify(
            {
                "received": "file",
                "filename": filename,
                "size": file_size,
                "note": "placeholder inference",
                "result": result,
            }
        )

    payload = request.get_json(silent=True)
    if payload is not None:
        processed_payload = preprocess_json(payload)
        result = inference_service.predict_from_json(processed_payload)
        top_level_keys = list(payload.keys()) if isinstance(payload, dict) else []

        return jsonify(
            {
                "received": "json",
                "keys": top_level_keys,
                "note": "placeholder inference",
                "result": result,
            }
        )

    return jsonify({"error": 'Provide either multipart field "file" or a JSON body.'}), 400
