import os

from app import create_app

app = create_app()


if __name__ == "__main__":
    ssl_context = "adhoc" if os.environ.get("FLASK_SSL_ADHOC") == "1" else None
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=ssl_context)

