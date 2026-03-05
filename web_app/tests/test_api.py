from pathlib import Path

import pytest

from app import create_app
from app.routes import api


@pytest.fixture
def app(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        }
    )
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_json(client):
    payload = {"sensor_id": "demo", "values": [0.1, 0.2, 0.3]}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200

    body = response.get_json()
    assert body["received"] == "json"
    assert set(body["keys"]) == set(payload.keys())
    assert body["note"] == "placeholder inference"


def test_prepare_stream_request_passes_fp16_to_keypoint_resolver(monkeypatch, tmp_path):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    (videos_dir / "demo.mp4").write_bytes(b"video")

    captured = {}

    def fake_ensure_keypoint_asset(*, model_name: str, relative_path: str, keypoint_precision: str) -> Path:
        captured["model_name"] = model_name
        captured["relative_path"] = relative_path
        captured["keypoint_precision"] = keypoint_precision
        return Path("/tmp/yolo11l-pose_fp16.engine")

    monkeypatch.setattr(api, "_list_available_test_videos", lambda: (videos_dir, ["demo.mp4"]))
    monkeypatch.setattr(api, "_ensure_keypoint_asset", fake_ensure_keypoint_asset)
    monkeypatch.setattr(api, "_resolve_classification_weights_path", lambda **_kwargs: Path("/tmp/tcn_best.pt"))
    monkeypatch.setattr(api, "_infer_keypoint_backend", lambda *_args, **_kwargs: "yolo")

    prepared = api._prepare_stream_request(
        {
            "classification_model": "tcn",
            "keypoint_model": "ultralytics-yolo11l",
            "keypoint_precision": "FP16",
            "video": "demo.mp4",
        }
    )

    assert captured["model_name"] == "ultralytics-yolo11l"
    assert captured["relative_path"] == "ultralytics/yolo11l-pose"
    assert captured["keypoint_precision"] == "FP16"
    assert prepared["keypoint_precision"] == "FP16"
    assert prepared["keypoint_model_path"].name == "yolo11l-pose_fp16.engine"


def test_prepare_stream_request_defaults_precision_to_fp32(monkeypatch, tmp_path):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    (videos_dir / "demo.mp4").write_bytes(b"video")

    captured = {}

    def fake_ensure_keypoint_asset(*, model_name: str, relative_path: str, keypoint_precision: str) -> Path:
        captured["keypoint_precision"] = keypoint_precision
        return Path("/tmp/yolo11l-pose.pt")

    monkeypatch.setattr(api, "_list_available_test_videos", lambda: (videos_dir, ["demo.mp4"]))
    monkeypatch.setattr(api, "_ensure_keypoint_asset", fake_ensure_keypoint_asset)
    monkeypatch.setattr(api, "_resolve_classification_weights_path", lambda **_kwargs: Path("/tmp/tcn_best.pt"))
    monkeypatch.setattr(api, "_infer_keypoint_backend", lambda *_args, **_kwargs: "yolo")

    prepared = api._prepare_stream_request(
        {
            "classification_model": "tcn",
            "keypoint_model": "ultralytics-yolo11l",
            "video": "demo.mp4",
        }
    )

    assert captured["keypoint_precision"] == "FP32"
    assert prepared["keypoint_precision"] == "FP32"


def test_ensure_keypoint_asset_fp16_missing_file_raises(monkeypatch, tmp_path):
    keypoint_root = tmp_path / "models" / "keypoint"
    (keypoint_root / "ultralytics" / "yolo11l-pose").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api, "_keypoint_models_root", lambda: keypoint_root)

    with pytest.raises(FileNotFoundError) as error:
        api._ensure_keypoint_asset(
            model_name="ultralytics-yolo11l",
            relative_path="ultralytics/yolo11l-pose",
            keypoint_precision="FP16",
        )

    assert "yolo11l-pose_fp16.engine" in str(error.value)
