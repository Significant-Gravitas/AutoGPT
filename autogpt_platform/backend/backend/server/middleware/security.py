import re
from typing import Set

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class SecurityHeadersMiddleware:
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
        "/api/blocks",
        "/api/v1/blocks",
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
        self.app = app
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

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pure ASGI middleware implementation for better performance than BaseHTTPMiddleware."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract path from scope
        path = scope["path"]

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                # Add security headers to the response
                headers = dict(message.get("headers", []))

                # Add general security headers (HTTP spec requires proper capitalization)
                headers[b"X-Content-Type-Options"] = b"nosniff"
                headers[b"X-Frame-Options"] = b"DENY"
                headers[b"X-XSS-Protection"] = b"1; mode=block"
                headers[b"Referrer-Policy"] = b"strict-origin-when-cross-origin"

                # Add noindex header for shared execution pages
                if "/public/shared" in path:
                    headers[b"X-Robots-Tag"] = b"noindex, nofollow"

                # Default: Disable caching for all endpoints
                # Only allow caching for explicitly permitted paths
                if not self.is_cacheable_path(path):
                    headers[b"Cache-Control"] = (
                        b"no-store, no-cache, must-revalidate, private"
                    )
                    headers[b"Pragma"] = b"no-cache"
                    headers[b"Expires"] = b"0"

                # Convert headers back to list format
                message["headers"] = list(headers.items())

            await send(message)

        await self.app(scope, receive, send_wrapper)
