"""Main Flask application entry point."""

import os
import socket
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from flask import Flask
from flask_cors import CORS

from database import init_database
from config import API_HOST, API_PORT, API_DEBUG, ENVIRONMENT


def port_is_in_use(host, port):
    """Return True when another process is already listening on the API port."""
    probe_host = "127.0.0.1" if host == "0.0.0.0" else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((probe_host, port)) == 0


def create_app():
    """Create and configure Flask application."""
    from routes.predict import predict_bp
    from routes.analytics import analytics_bp
    from routes.explain import explain_bp

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
    if port_is_in_use(API_HOST, API_PORT):
        print(f"Backend is already running at http://127.0.0.1:{API_PORT}")
        print("Stop the existing Python backend process first if you want to restart it.")
        sys.exit(0)

    app = create_app()
    print(f"Backend running at http://127.0.0.1:{API_PORT}")
    if ENVIRONMENT == 'production':
        try:
            from waitress import serve
            serve(app, host=API_HOST, port=API_PORT)
        except ImportError:
            app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG, use_reloader=False)
    else:
        app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG, use_reloader=False)
