import re
from typing import Set

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to responses, with cache control
    disabled by default for all endpoints except those explicitly allowed.
    """

    CACHEABLE_PATHS: Set[str] = {
        # Static assets
        "/static",
        "/_next/static",
        "/assets",
        "/images",
        "/css",
        "/js",
        "/fonts",
        # Public API endpoints
        "/api/health",
        "/api/v1/health",
        "/api/status",
        # Public store/marketplace pages (read-only)
        "/api/store/agents",
        "/api/v1/store/agents",
        "/api/store/categories",
        "/api/v1/store/categories",
        "/api/store/featured",
        "/api/v1/store/featured",
        # Public graph templates (read-only, no user data)
        "/api/graphs/templates",
        "/api/v1/graphs/templates",
        # Documentation endpoints
        "/api/docs",
        "/api/v1/docs",
        "/docs",
        "/swagger",
        "/openapi.json",
        # Favicon and manifest
        "/favicon.ico",
        "/manifest.json",
        "/robots.txt",
        "/sitemap.xml",
    }

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        # Compile regex patterns for wildcard matching
        self.cacheable_patterns = [
            re.compile(pattern.replace("*", "[^/]+"))
            for pattern in self.CACHEABLE_PATHS
            if "*" in pattern
        ]
        self.exact_paths = {path for path in self.CACHEABLE_PATHS if "*" not in path}

    def is_cacheable_path(self, path: str) -> bool:
        """Check if the given path is allowed to be cached."""
        # Check exact matches first
        for cacheable_path in self.exact_paths:
            if path.startswith(cacheable_path):
                return True

        # Check pattern matches
        for pattern in self.cacheable_patterns:
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

        # Default: Disable caching for all endpoints
        # Only allow caching for explicitly permitted paths
        if not self.is_cacheable_path(request.url.path):
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, private"
            )
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response
