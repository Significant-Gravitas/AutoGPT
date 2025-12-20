"""
Authentication module for the AutoGPT Platform.

This module provides FastAPI-based authentication supporting:
- Email/password authentication with bcrypt hashing
- Google OAuth authentication
- JWT token management (access + refresh tokens)
"""

from .routes import router as auth_router
from .service import AuthService

__all__ = ["auth_router", "AuthService"]
