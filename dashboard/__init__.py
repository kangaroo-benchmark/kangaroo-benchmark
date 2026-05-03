"""Dashboard package exposing FastAPI app factory."""

from .server import create_app

__all__ = ["create_app"]
