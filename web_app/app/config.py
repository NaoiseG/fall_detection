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

# TODO: Replace each value with the actual relative checkpoint path under
# .\models\classification\... for your project.
CLASSIFICATION_MODELS = {
    "tcn": "tcn/tcn_best.pt",
    "cnn-lstm": "cnnlstm/cnnlstm_best.pt",
    "st-gcn": "stgcn/stgcn_best.pt",
    "gru": "gru/gru_best.pt",
    "motionbert": "MotionBERT/checkpoint/action/FT_MB_release_MB_ft_UPFall_xsub/best_epoch.bin",
}

# TODO: Replace each value with the actual relative checkpoint path under
# .\models\keypoint\... for your project.
KEYPOINT_MODELS = {
    "ultralytics-yolo11n": "ultralytics/yolo11n-pose.pt",
    "ultralytics-yolo11s": "ultralytics/yolo11s-pose.pt",
    "ultralytics-yolo11m": "ultralytics/yolo11m-pose.pt",
    "ultralytics-yolo11l": "ultralytics/yolo11l-pose.pt",
    "ultralytics-yolo11x": "ultralytics/yolo11x-pose.pt",
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
