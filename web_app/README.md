# Fall Detection Web App (Flask Scaffold)

Minimal Flask scaffold using the app factory pattern.

## Quickstart

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
python run.py
```

4. Open:

- `http://127.0.0.1:5000/`

## API Examples

Health check:

```powershell
curl http://127.0.0.1:5000/api/health
```

Predict from JSON:

```powershell
curl -X POST http://127.0.0.1:5000/api/predict `
  -H "Content-Type: application/json" `
  -d "{\"event\":\"fall_candidate\",\"score\":0.5}"
```

Predict from file:

```powershell
curl -X POST http://127.0.0.1:5000/api/predict -F "file=@README.md"
```

## Configuration

Set environment variables in your shell, or add instance overrides in `instance/config.py`.

- `SECRET_KEY` (default: `dev`)
- `UPLOAD_FOLDER` (default: `app/static/uploads`)
- `MAX_CONTENT_LENGTH` in bytes (default: `209715200`, i.e. 200MB)

Reference values are provided in `.env.example`.

