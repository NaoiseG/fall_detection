from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from app.services.inference_service import InferenceService
from app.services.preprocessing import preprocess_file, preprocess_json

api_bp = Blueprint("api", __name__, url_prefix="/api")
inference_service = InferenceService()
inference_service.load()


@api_bp.get("/health")
def health():
    return jsonify({"status": "ok"})


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

