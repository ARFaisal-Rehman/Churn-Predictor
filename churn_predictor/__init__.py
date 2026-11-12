from flask import Flask

from .config import Config
from .routes import bp
from .services.analysis_service import AnalysisService


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config.from_object(Config)
    app.extensions["analysis_service"] = AnalysisService(app.config)
    app.register_blueprint(bp)
    return app
