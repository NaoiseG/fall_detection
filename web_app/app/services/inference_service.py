class InferenceService:
    """Placeholder inference service."""

    def load(self):
        """Load model artifacts (no-op for now)."""
        return None

    def predict_from_file(self, path):
        """Return dummy inference output for a file path."""
        return {"prediction": "dummy", "source": "file", "path": str(path)}

    def predict_from_json(self, data):
        """Return dummy inference output for JSON payload."""
        return {"prediction": "dummy", "source": "json", "payload_type": type(data).__name__}

