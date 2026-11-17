"""FastAPI entry point and server initialization for TabSynth Analytics."""

from api.main import app, create_app
from api.routes import api_v1_router as router

__all__ = ["app", "create_app", "router"]
