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

    @app.get("/public/data")
    def public_endpoint():
        return {"public": "data"}

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


def test_sensitive_endpoints_have_cache_control_headers(client):
    """Test that sensitive endpoints have proper cache control headers."""
    sensitive_endpoints = [
        "/api/auth/user",
        "/api/v1/integrations/oauth/google",
        "/api/graphs/123/execute",
        "/api/integrations/credentials",
    ]

    for endpoint in sensitive_endpoints:
        response = client.get(endpoint)

        # Check cache control headers
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


def test_public_endpoints_dont_have_cache_control_headers(client):
    """Test that public endpoints don't have cache control headers."""
    response = client.get("/public/data")

    # Should NOT have cache control headers
    assert (
        "Cache-Control" not in response.headers
        or "no-store" not in response.headers.get("Cache-Control", "")
    )
    assert (
        "Pragma" not in response.headers or response.headers.get("Pragma") != "no-cache"
    )
    assert "Expires" not in response.headers or response.headers.get("Expires") != "0"

    # Should still have general security headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_is_sensitive_path_detection():
    """Test the path detection logic."""
    middleware = SecurityHeadersMiddleware(Starlette())

    # Test exact matches
    assert middleware.is_sensitive_path("/api/auth/user")
    assert middleware.is_sensitive_path("/api/v1/integrations/oauth/callback")
    assert middleware.is_sensitive_path("/api/integrations/credentials/123")

    # Test pattern matches
    assert middleware.is_sensitive_path("/api/graphs/abc123/execute")
    assert middleware.is_sensitive_path("/api/v1/graphs/def456/execute")
    assert middleware.is_sensitive_path("/api/store/xyz/submissions")

    # Test non-sensitive paths
    assert not middleware.is_sensitive_path("/api/public/data")
    assert not middleware.is_sensitive_path("/health")
    assert not middleware.is_sensitive_path("/docs")


def test_wildcard_pattern_matching():
    """Test that wildcard patterns work correctly."""
    middleware = SecurityHeadersMiddleware(Starlette())

    # Test graph execution patterns
    assert middleware.is_sensitive_path("/api/graphs/12345/execute")
    assert middleware.is_sensitive_path("/api/v1/graphs/abcdef/execute")

    # Test store submission patterns
    assert middleware.is_sensitive_path("/api/store/test-store/submissions")

    # Test graph export patterns
    assert middleware.is_sensitive_path("/api/graphs/12345/export")
    assert middleware.is_sensitive_path("/api/v1/graphs/abcdef/export")

    # Test non-matching patterns
    assert not middleware.is_sensitive_path("/api/graphs/12345/details")
    assert not middleware.is_sensitive_path("/api/store/test-store/public")
