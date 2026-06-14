"""Main Flask application entry point."""

from flask import Flask
from flask_cors import CORS

from routes.predict import predict_bp
from routes.analytics import analytics_bp
from routes.explain import explain_bp
from database import init_database
from config import API_HOST, API_PORT, API_DEBUG


def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)
    
    # Enable CORS for Streamlit
    CORS(app)
    
    # Initialize database
    init_database()
    
    # Register blueprints
    app.register_blueprint(predict_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(explain_bp)
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)