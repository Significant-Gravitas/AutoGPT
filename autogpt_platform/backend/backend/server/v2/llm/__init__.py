"""LLM registry API (public + admin)."""

from .admin_routes import router as admin_router
from .routes import router

__all__ = ["router", "admin_router"]
