"""Routes module initialization."""

from routes.predict import predict_bp
from routes.analytics import analytics_bp
from routes.explain import explain_bp

__all__ = ['predict_bp', 'analytics_bp', 'explain_bp']