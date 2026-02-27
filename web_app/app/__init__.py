from flask import Flask, jsonify, request

from app.config import Config
from app.extensions import init_extensions
from app.routes.api import api_bp
from app.routes.main import main_bp


def create_app(config_object=None):
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_object(Config)
    app.config.from_pyfile("config.py", silent=True)

    if config_object:
        if isinstance(config_object, dict):
            app.config.from_mapping(config_object)
        else:
            app.config.from_object(config_object)

    Config.init_app(app)
    init_extensions(app)

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)

    @app.errorhandler(400)
    def handle_bad_request(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "bad_request", "message": str(error)}), 400
        return "Bad Request", 400

    @app.errorhandler(500)
    def handle_internal_error(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "internal_server_error", "message": "Unexpected server error."}), 500
        return "Internal Server Error", 500

    return app

