"""Tests for SecurityHeadersMiddleware."""

import fastapi
import fastapi.testclient
import pytest

from backend.api.middleware.security import SecurityHeadersMiddleware

app = fastapi.FastAPI()
app.add_middleware(SecurityHeadersMiddleware)


@app.get("/api/private/data")
def private_endpoint():
    return {"secret": "value"}


@app.get("/api/health")
def health_endpoint():
    return {"status": "ok"}


@app.get("/static/style.css")
def static_asset():
    return {"css": "body{}"}


@app.get("/api/store/agents")
def store_agents():
    return {"agents": []}


@app.get("/api/public/shared/exec123")
def shared_execution():
    return {"result": "shared"}


client = fastapi.testclient.TestClient(app)


EXPECTED_SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "strict-transport-security": "max-age=31536000; includeSubDomains",
    "referrer-policy": "strict-origin-when-cross-origin",
    "permissions-policy": (
        "camera=(), microphone=(), geolocation=(), "
        "payment=(), usb=(), interest-cohort=()"
    ),
}


class TestSecurityHeaders:
    """Verify that all standard security headers are present on responses."""

    def test_security_headers_on_private_endpoint(self):
        response = client.get("/api/private/data")
        assert response.status_code == 200
        for header, expected_value in EXPECTED_SECURITY_HEADERS.items():
            assert response.headers.get(header) == expected_value, (
                f"Header {header!r} expected {expected_value!r}, "
                f"got {response.headers.get(header)!r}"
            )

    def test_security_headers_on_cacheable_endpoint(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        for header, expected_value in EXPECTED_SECURITY_HEADERS.items():
            assert response.headers.get(header) == expected_value

    def test_xss_protection_header(self):
        response = client.get("/api/private/data")
        assert response.headers.get("x-xss-protection") == "1; mode=block"


class TestCacheControl:
    """Verify cache-control behaviour for cacheable and non-cacheable paths."""

    def test_non_cacheable_path_has_no_store(self):
        response = client.get("/api/private/data")
        assert response.status_code == 200
        cache_control = response.headers.get("cache-control")
        assert cache_control is not None
        assert "no-store" in cache_control
        assert "no-cache" in cache_control
        assert "must-revalidate" in cache_control
        assert "private" in cache_control
        assert response.headers.get("pragma") == "no-cache"
        assert response.headers.get("expires") == "0"

    def test_cacheable_health_endpoint_no_forced_no_store(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        cache_control = response.headers.get("cache-control", "")
        # The middleware should NOT force no-store on cacheable paths
        assert "no-store" not in cache_control

    def test_cacheable_static_path_no_forced_no_store(self):
        response = client.get("/static/style.css")
        assert response.status_code == 200
        cache_control = response.headers.get("cache-control", "")
        assert "no-store" not in cache_control

    def test_cacheable_store_agents_no_forced_no_store(self):
        response = client.get("/api/store/agents")
        assert response.status_code == 200
        cache_control = response.headers.get("cache-control", "")
        assert "no-store" not in cache_control

    def test_pragma_and_expires_absent_on_cacheable_path(self):
        response = client.get("/api/health")
        # Pragma and Expires are only set for non-cacheable paths
        assert response.headers.get("pragma") is None
        assert response.headers.get("expires") is None


class TestContentSecurityPolicy:
    """Verify CSP header is set with default-src 'none' for JSON API responses."""

    def test_csp_header_on_json_response(self):
        response = client.get("/api/private/data")
        assert response.status_code == 200
        csp = response.headers.get("content-security-policy")
        assert csp is not None
        assert "default-src 'none'" in csp

    def test_csp_header_on_health_endpoint(self):
        response = client.get("/api/health")
        csp = response.headers.get("content-security-policy")
        assert csp is not None
        assert "default-src 'none'" in csp


class TestRobotsTag:
    """Verify X-Robots-Tag for shared execution pages."""

    def test_shared_execution_has_noindex(self):
        response = client.get("/api/public/shared/exec123")
        assert response.status_code == 200
        robots = response.headers.get("x-robots-tag")
        assert robots is not None
        assert "noindex" in robots
        assert "nofollow" in robots

    def test_regular_endpoint_no_robots_tag(self):
        response = client.get("/api/private/data")
        assert response.headers.get("x-robots-tag") is None


class TestIsCacheablePath:
    """Unit tests for the is_cacheable_path method."""

    @pytest.fixture()
    def middleware(self):
        return SecurityHeadersMiddleware(app=None)

    @pytest.mark.parametrize(
        "path",
        [
            "/static/logo.png",
            "/api/health",
            "/api/v1/health",
            "/api/blocks",
            "/api/store/agents",
            "/api/v1/store/categories",
            "/docs",
            "/openapi.json",
            "/favicon.ico",
            "/robots.txt",
        ],
    )
    def test_cacheable_paths(self, middleware, path):
        assert middleware.is_cacheable_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/api/graphs/123",
            "/api/users/me",
            "/api/settings",
            "/api/auth/login",
            "/api/integrations",
        ],
    )
    def test_non_cacheable_paths(self, middleware, path):
        assert middleware.is_cacheable_path(path) is False
