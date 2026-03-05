import pytest

from app import create_app


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
