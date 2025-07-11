import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.applications import Starlette

from backend.server.middleware.security import SecurityHeadersMiddleware


@pytest.fixture
def app():
    """Create a test FastAPI app with security middleware."""
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/api/auth/user")
    def get_user():
        return {"user": "test"}

    @app.get("/api/v1/integrations/oauth/google")
    def oauth_endpoint():
        return {"oauth": "data"}

    @app.get("/api/graphs/123/execute")
    def execute_graph():
        return {"execution": "data"}

    @app.get("/api/integrations/credentials")
    def get_credentials():
        return {"credentials": "sensitive"}

    @app.get("/api/store/agents")
    def store_agents():
        return {"agents": "public list"}

    @app.get("/api/health")
    def health_check():
        return {"status": "ok"}

    @app.get("/static/logo.png")
    def static_file():
        return {"static": "content"}

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


def test_non_cacheable_endpoints_have_cache_control_headers(client):
    """Test that non-cacheable endpoints (most endpoints) have proper cache control headers."""
    non_cacheable_endpoints = [
        "/api/auth/user",
        "/api/v1/integrations/oauth/google",
        "/api/graphs/123/execute",
        "/api/integrations/credentials",
    ]

    for endpoint in non_cacheable_endpoints:
        response = client.get(endpoint)

        # Check cache control headers are present (default behavior)
        assert (
            response.headers["Cache-Control"]
            == "no-store, no-cache, must-revalidate, private"
        )
        assert response.headers["Pragma"] == "no-cache"
        assert response.headers["Expires"] == "0"

        # Check general security headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_cacheable_endpoints_dont_have_cache_control_headers(client):
    """Test that explicitly cacheable endpoints don't have restrictive cache control headers."""
    cacheable_endpoints = [
        "/api/store/agents",
        "/api/health",
        "/static/logo.png",
    ]

    for endpoint in cacheable_endpoints:
        response = client.get(endpoint)

        # Should NOT have restrictive cache control headers
        assert (
            "Cache-Control" not in response.headers
            or "no-store" not in response.headers.get("Cache-Control", "")
        )
        assert (
            "Pragma" not in response.headers
            or response.headers.get("Pragma") != "no-cache"
        )
        assert (
            "Expires" not in response.headers or response.headers.get("Expires") != "0"
        )

        # Should still have general security headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_is_cacheable_path_detection():
    """Test the path detection logic."""
    middleware = SecurityHeadersMiddleware(Starlette())

    # Test cacheable paths (allow list)
    assert middleware.is_cacheable_path("/api/health")
    assert middleware.is_cacheable_path("/api/v1/health")
    assert middleware.is_cacheable_path("/static/image.png")
    assert middleware.is_cacheable_path("/api/store/agents")
    assert middleware.is_cacheable_path("/docs")
    assert middleware.is_cacheable_path("/favicon.ico")

    # Test non-cacheable paths (everything else)
    assert not middleware.is_cacheable_path("/api/auth/user")
    assert not middleware.is_cacheable_path("/api/v1/integrations/oauth/callback")
    assert not middleware.is_cacheable_path("/api/integrations/credentials/123")
    assert not middleware.is_cacheable_path("/api/graphs/abc123/execute")
    assert not middleware.is_cacheable_path("/api/store/xyz/submissions")


def test_path_prefix_matching():
    """Test that path prefix matching works correctly."""
    middleware = SecurityHeadersMiddleware(Starlette())

    # Test that paths starting with cacheable prefixes are cacheable
    assert middleware.is_cacheable_path("/static/css/style.css")
    assert middleware.is_cacheable_path("/static/js/app.js")
    assert middleware.is_cacheable_path("/assets/images/logo.png")
    assert middleware.is_cacheable_path("/_next/static/chunks/main.js")

    # Test that other API paths are not cacheable by default
    assert not middleware.is_cacheable_path("/api/users/profile")
    assert not middleware.is_cacheable_path("/api/v1/private/data")
    assert not middleware.is_cacheable_path("/api/billing/subscription")
