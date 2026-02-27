from pathlib import Path
import subprocess
import sys

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from app.config import (
    CLASSIFICATION_MODELS,
    KEYPOINT_MODELS,
    VIDEO_EXTENSIONS,
    get_test_videos_dir,
    get_web_app_root,
)
from app.services.inference_service import InferenceService
from app.services.preprocessing import preprocess_file, preprocess_json

api_bp = Blueprint("api", __name__, url_prefix="/api")
inference_service = InferenceService()
inference_service.load()


def _json_error(message, status_code, command="", returncode=None, stdout="", stderr=""):
    return (
        jsonify(
            {
                "ok": False,
                "command": command,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr or message,
                "error": message,
            }
        ),
        status_code,
    )


def _list_available_test_videos():
    test_videos_dir = get_test_videos_dir()
    if not test_videos_dir.exists() or not test_videos_dir.is_dir():
        raise FileNotFoundError(f"Test video directory does not exist: {test_videos_dir}")

    videos = sorted(
        file_path.name
        for file_path in test_videos_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return test_videos_dir, videos


def _is_safe_video_filename(video_name):
    if not isinstance(video_name, str):
        return False

    candidate = video_name.strip()
    if not candidate:
        return False

    if "/" in candidate or "\\" in candidate or ".." in candidate:
        return False

    return Path(candidate).name == candidate


def _windows_rel_path(base_prefix, relative_path):
    cleaned = str(relative_path).replace("/", "\\")
    return f"{base_prefix}\\{cleaned}"


def _keypoint_models_root() -> Path:
    return (get_web_app_root() / "models" / "keypoint").resolve()


def _ensure_keypoint_weights(relative_path: str) -> Path:
    keypoint_root = _keypoint_models_root()
    weights_rel_path = Path(relative_path)
    if weights_rel_path.is_absolute():
        raise ValueError("Keypoint model path must be relative.")

    weights_path = (keypoint_root / weights_rel_path).resolve()
    try:
        weights_path.relative_to(keypoint_root)
    except ValueError as error:
        raise ValueError("Invalid keypoint model path.") from error

    if weights_path.is_file():
        return weights_path

    weights_path.parent.mkdir(parents=True, exist_ok=True)

    # Use Ultralytics' official asset downloader when a known model file is missing.
    from ultralytics.utils.downloads import attempt_download_asset

    attempt_download_asset(str(weights_path))

    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing keypoint model weights: {weights_path}")

    return weights_path


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


@api_bp.post("/run_inference")
def run_inference():
    command_string = ""

    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _json_error("Invalid JSON body. Expected an object.", 400)

        classification_model = payload.get("classification_model")
        keypoint_model = payload.get("keypoint_model")
        video_name = payload.get("video")

        if classification_model not in CLASSIFICATION_MODELS:
            return _json_error("Invalid classification model selection.", 400)
        if keypoint_model not in KEYPOINT_MODELS:
            return _json_error("Invalid keypoint model selection.", 400)
        if not _is_safe_video_filename(video_name):
            return _json_error("Invalid video value. Expected a filename without path separators.", 400)

        video_name = video_name.strip()

        try:
            test_videos_dir, available_videos = _list_available_test_videos()
        except (FileNotFoundError, OSError) as error:
            return _json_error(str(error), 500)

        if video_name not in available_videos:
            return _json_error("Selected video is not available in the test video directory.", 400)

        resolved_video_path = (test_videos_dir / video_name).resolve()
        try:
            resolved_video_path.relative_to(test_videos_dir)
        except ValueError:
            return _json_error("Invalid video path resolution.", 400)
        if not resolved_video_path.is_file():
            return _json_error("Selected video file does not exist.", 400)

        keypoint_model_path = KEYPOINT_MODELS[keypoint_model]
        try:
            _ensure_keypoint_weights(keypoint_model_path)
        except Exception as error:
            return _json_error(f"Failed to prepare keypoint model weights: {error}", 500)

        command = [
            sys.executable,
            "-m",
            "inference.inference_on_video",
            "--video",
            f"..\\Datasets\\test_vids\\{video_name}",
            "--keypoint-model",
            _windows_rel_path(".\\models\\keypoint", keypoint_model_path),
            "--model",
            _windows_rel_path(".\\models\\classification", CLASSIFICATION_MODELS[classification_model]),
            "--T",
            "64",
            "--stride",
            "32",
            "--normalize-mode",
            "paper_rp",
            "--rp-center-mode",
            "pixel",
            "--rp-img-w",
            "640",
            "--rp-img-h",
            "480",
            "--missing-mode",
            "zeros_only",
            "--interp-mode",
            "paper_group_linear",
            "--interp-group",
            "100",
        ]
        command_string = subprocess.list2cmdline(command)

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=get_web_app_root(),
                timeout=600,
            )
        except subprocess.TimeoutExpired as error:
            timeout_message = "Inference command timed out after 600 seconds."
            stderr = timeout_message if not error.stderr else f"{timeout_message}\n{error.stderr}"
            return (
                jsonify(
                    {
                        "ok": False,
                        "command": command_string,
                        "returncode": -1,
                        "stdout": error.stdout or "",
                        "stderr": stderr,
                    }
                ),
                500,
            )
        except OSError as error:
            return _json_error(f"Failed to execute inference command: {error}", 500, command=command_string)

        return jsonify(
            {
                "ok": completed.returncode == 0,
                "command": command_string,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    except Exception as error:
        return _json_error(f"Unexpected inference error: {error}", 500, command=command_string)


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
