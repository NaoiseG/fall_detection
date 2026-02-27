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

