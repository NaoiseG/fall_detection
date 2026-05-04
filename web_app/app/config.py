import os
from pathlib import Path


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Relative model directories under .\models\classification\...
CLASSIFICATION_MODELS = {
    "tcn": "tcn",
    "cnn-lstm": "cnnlstm",
    "st-gcn": "stgcn",
    "motionbert": "MotionBERT",
}

# Expected weight filename for each classification model (searched recursively
# inside the keypoint-conditioned subdirectory).
CLASSIFICATION_WEIGHT_FILENAMES = {
    "tcn": "tcn_best.pt",
    "cnn-lstm": "cnnlstm_best.pt",
    "st-gcn": "paper_stgcn_best.pt",
    "motionbert": "best_epoch.bin",
}

# Relative model directories under .\models\keypoint\...
KEYPOINT_MODELS = {
    "ultralytics-yolo11n": "ultralytics/yolo11n-pose",
    "ultralytics-yolo11s": "ultralytics/yolo11s-pose",
    "ultralytics-yolo11m": "ultralytics/yolo11m-pose",
    "ultralytics-yolo11l": "ultralytics/yolo11l-pose",
    "ultralytics-yolo11x": "ultralytics/yolo11x-pose",
    "vitpose-base": "ViTPose",
    "alphapose-fastpose": "AlphaPose",
}

# The keypoint model determines which subdirectory to search under each
# classification model directory.
KEYPOINT_TO_CLASSIFICATION_SUBDIR = {
    "ultralytics-yolo11n": "yolo11n-pose",
    "ultralytics-yolo11s": "yolo11s-pose",
    "ultralytics-yolo11m": "yolo11m-pose",
    "ultralytics-yolo11l": "yolo11l-pose",
    "ultralytics-yolo11x": "yolo11x-pose",
    "vitpose-base": "vitpose",
    "alphapose-fastpose": "alphapose",
}

# Expected keypoint checkpoint filename for each Ultralytics model directory.
KEYPOINT_YOLO_WEIGHT_FILENAMES = {
    "ultralytics-yolo11n": "yolo11n-pose.pt",
    "ultralytics-yolo11s": "yolo11s-pose.pt",
    "ultralytics-yolo11m": "yolo11m-pose.pt",
    "ultralytics-yolo11l": "yolo11l-pose.pt",
    "ultralytics-yolo11x": "yolo11x-pose.pt",
}

# Supported precision dropdown values in the web app.
DEFAULT_KEYPOINT_PRECISION = "FP32"
KEYPOINT_PRECISIONS = ("FP32", "FP16")

# Optional FP16 engine filenames for Ultralytics keypoint models.
KEYPOINT_YOLO_FP16_WEIGHT_FILENAMES = {
    "ultralytics-yolo11n": "yolo11n-pose_fp16.engine",
    "ultralytics-yolo11s": "yolo11s-pose_fp16.engine",
    "ultralytics-yolo11m": "yolo11m-pose_fp16.engine",
    "ultralytics-yolo11l": "yolo11l-pose_fp16.engine",
    "ultralytics-yolo11x": "yolo11x-pose_fp16.engine",
}


def get_repo_root() -> Path:
    # app/config.py -> <repo>/web_app/app/config.py, so parents[2] is <repo>.
    return Path(__file__).resolve().parents[2]


def get_web_app_root() -> Path:
    # app/config.py -> <repo>/web_app/app/config.py, so parents[1] is <repo>/web_app.
    return Path(__file__).resolve().parents[1]


def get_test_videos_dir() -> Path:
    # Test videos live under <repo_root>\Datasets\test_vids.
    return (get_repo_root() / "Datasets" / "test_vids").resolve()


def get_saved_outputs_dir() -> Path:
    return (get_web_app_root() / "saved_outputs").resolve()


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev")
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "app/static/uploads")
    MAX_CONTENT_LENGTH = _env_int("MAX_CONTENT_LENGTH", 200 * 1024 * 1024)

    @staticmethod
    def init_app(app):
        upload_folder = Path(app.config["UPLOAD_FOLDER"])
        if not upload_folder.is_absolute():
            upload_folder = Path(app.root_path).parent / upload_folder

        upload_folder.mkdir(parents=True, exist_ok=True)
        app.config["UPLOAD_FOLDER"] = str(upload_folder)
