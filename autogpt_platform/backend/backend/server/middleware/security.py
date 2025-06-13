import re
from typing import Set

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to responses, including cache control
    for sensitive endpoints to prevent caching of sensitive data.
    """

    SENSITIVE_PATHS: Set[str] = {
        # Authentication endpoints
        "/api/auth",
        "/api/v1/auth",
        # OAuth endpoints
        "/api/integrations/oauth",
        "/api/v1/integrations/oauth",
        # User credentials and sensitive data
        "/api/integrations/credentials",
        "/api/v1/integrations/credentials",
        "/api/integrations",
        "/api/v1/integrations",
        # User profile and sensitive user data
        "/api/auth/user",
        "/api/v1/auth/user",
        "/api/users",
        "/api/v1/users",
        # Credit and billing information
        "/api/credits",
        "/api/v1/credits",
        # Graph execution (may contain credentials)
        "/api/graphs/*/execute",
        "/api/v1/graphs/*/execute",
        # Store admin and sensitive operations
        "/api/store/admin",
        "/api/store/*/submissions",
        # Library operations (may contain user data)
        "/api/library",
        "/api/v2/library",
        # Otto chat (contains conversation data)
        "/api/otto",
        "/api/v2/otto",
        # External API endpoints (may contain sensitive integrations)
        "/external-api",
        # Email endpoints (may contain personal data)
        "/api/email",
        # API key management
        "/api/v1/api_keys",
        # Graph definitions (may contain embedded credentials) - but allow public read access for basic metadata
        "/api/graphs/*/export",
        "/api/v1/graphs/*/export",
    }

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        # Compile regex patterns for wildcard matching
        self.sensitive_patterns = [
            re.compile(pattern.replace("*", "[^/]+"))
            for pattern in self.SENSITIVE_PATHS
            if "*" in pattern
        ]
        self.exact_paths = {path for path in self.SENSITIVE_PATHS if "*" not in path}

    def is_sensitive_path(self, path: str) -> bool:
        """Check if the given path should have cache protection."""
        # Check exact matches first
        for sensitive_path in self.exact_paths:
            if path.startswith(sensitive_path):
                return True

        # Check pattern matches
        for pattern in self.sensitive_patterns:
            if pattern.match(path):
                return True

        return False

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        # Add general security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Add cache control headers for sensitive endpoints
        if self.is_sensitive_path(request.url.path):
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, private"
            )
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response
