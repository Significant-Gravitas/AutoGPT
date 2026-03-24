"""Tests for get_user_id_from_request in the rate-limit middleware."""

import base64
import json

import pytest
from starlette.requests import Request

from backend.api.middleware.rate_limit import get_user_id_from_request


def _make_request(
    ip: str = "127.0.0.1",
    headers: dict[str, str] | None = None,
) -> Request:
    """Build a minimal Starlette Request with the given client IP and headers."""
    raw_headers: list[tuple[bytes, bytes]] = [
        (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": raw_headers,
        "client": (ip, 0),
    }
    return Request(scope)


def _make_jwt(payload: dict, header: dict | None = None) -> str:
    """Return a fake JWT (header.payload.signature) with the given payload."""
    header = header or {"alg": "RS256", "typ": "JWT"}
    h = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    p = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{h}.{p}.fakesignature"


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestGetUserIdFromRequest:
    """Tests for get_user_id_from_request."""

    def test_returns_ip_when_no_auth_header(self):
        request = _make_request(ip="10.0.0.1")
        assert get_user_id_from_request(request) == "10.0.0.1"

    def test_returns_sub_and_ip_for_valid_jwt(self):
        token = _make_jwt({"sub": "user-abc-123"})
        request = _make_request(
            ip="192.168.1.5",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert get_user_id_from_request(request) == "user-abc-123:192.168.1.5"

    def test_returns_ip_for_malformed_jwt(self):
        request = _make_request(
            ip="10.0.0.2",
            headers={"Authorization": "Bearer not-a-real-jwt"},
        )
        assert get_user_id_from_request(request) == "10.0.0.2"

    def test_returns_ip_when_jwt_has_no_sub(self):
        token = _make_jwt({"aud": "some-audience", "iss": "some-issuer"})
        request = _make_request(
            ip="10.0.0.3",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert get_user_id_from_request(request) == "10.0.0.3"

    def test_spoofed_sub_from_different_ips_creates_different_keys(self):
        """A forged token reusing the same sub must not collide with the real
        user when requests originate from different IPs."""
        token = _make_jwt({"sub": "victim-user-id"})
        real_request = _make_request(
            ip="1.2.3.4",
            headers={"Authorization": f"Bearer {token}"},
        )
        spoofed_request = _make_request(
            ip="5.6.7.8",
            headers={"Authorization": f"Bearer {token}"},
        )
        real_key = get_user_id_from_request(real_request)
        spoofed_key = get_user_id_from_request(spoofed_request)

        assert real_key == "victim-user-id:1.2.3.4"
        assert spoofed_key == "victim-user-id:5.6.7.8"
        assert real_key != spoofed_key
