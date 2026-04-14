"""Tests for Agent Discovery Protocol client."""

import json
import time
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from autogpt.utils.agent_discovery import (
    DiscoveryResult,
    _cache,
    discover_services,
)

SAMPLE_DISCOVERY = {
    "agent_discovery_version": "0.1",
    "domain": "example.com",
    "services": [
        {
            "name": "memory",
            "description": "Persistent memory",
            "endpoint": "https://example.com/api/memory",
            "auth": "bearer",
            "governance": "none",
            "free_tier": True,
        },
        {
            "name": "identity",
            "description": "Agent identity",
            "endpoint": "https://example.com/api/register",
            "auth": "bearer",
            "governance": "sift_lite",
            "free_tier": True,
        },
    ],
    "trust": {
        "verification_url": "https://example.com/verify",
    },
}


class TestDiscoveryResult:
    def test_list_services(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        assert sorted(result.list_services()) == [
            "identity",
            "memory",
        ]

    def test_has_service(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        assert result.has_service("memory") is True
        assert result.has_service("nonexistent") is False

    def test_get_service(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        memory = result.get_service("memory")
        assert memory is not None
        assert memory["endpoint"] == (
            "https://example.com/api/memory"
        )

    def test_get_service_missing(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        assert result.get_service("nonexistent") is None

    def test_domain(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        assert result.domain == "example.com"

    def test_version(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        assert result.version == "0.1"

    def test_trust(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        assert "verification_url" in result.trust

    def test_repr(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        r = repr(result)
        assert "example.com" in r
        assert "memory" in r

    def test_deepcopy_isolation(self):
        """Mutating returned service doesn't poison cache."""
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        svc = result.get_service("memory")
        svc["endpoint"] = "MUTATED"
        # Original should be unchanged
        original = result.get_service("memory")
        assert original["endpoint"] == (
            "https://example.com/api/memory"
        )

    def test_services_property_isolation(self):
        """Mutating .services dict doesn't affect internals."""
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        svcs = result.services
        svcs["memory"]["endpoint"] = "MUTATED"
        assert result.get_service("memory")["endpoint"] == (
            "https://example.com/api/memory"
        )


class TestSSRFValidation:
    def test_rejects_ip_literal(self):
        with pytest.raises(
            ValueError, match="invalid domain"
        ):
            discover_services("127.0.0.1")

    def test_rejects_ip_with_port(self):
        with pytest.raises(
            ValueError, match="invalid domain"
        ):
            discover_services("127.0.0.1:8080")

    def test_rejects_localhost(self):
        """localhost fails FQDN validation (no TLD)."""
        with pytest.raises(
            ValueError, match="invalid domain"
        ):
            discover_services("localhost")

    def test_rejects_empty(self):
        with pytest.raises(
            ValueError, match="non-empty"
        ):
            discover_services("")

    def test_rejects_scheme_injection(self):
        with pytest.raises(
            ValueError, match="invalid domain"
        ):
            discover_services("http://evil.com")

    def test_rejects_userinfo_injection(self):
        with pytest.raises(
            ValueError, match="invalid domain"
        ):
            discover_services("admin:pass@evil.com")

    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_rejects_private_ip_resolution(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 6, "", ("10.0.0.1", 443))
        ]
        with pytest.raises(
            ValueError, match="blocked address"
        ):
            discover_services("evil.example.com")

    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_rejects_metadata_ip(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 6, "", ("169.254.169.254", 443))
        ]
        with pytest.raises(
            ValueError, match="blocked address"
        ):
            discover_services("metadata.example.com")

    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_rejects_loopback_v6(self, mock_dns):
        mock_dns.return_value = [
            (10, 1, 6, "", ("::1", 443, 0, 0))
        ]
        with pytest.raises(
            ValueError, match="blocked address"
        ):
            discover_services("evil.example.com")


class TestRedirectProtection:
    """Verify 3xx redirects are blocked (SSRF bypass)."""

    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_redirect_blocked(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        _cache.clear()

        with patch(
            "autogpt.utils.agent_discovery.urllib.request"
            ".build_opener"
        ) as mock_opener:
            mock_resp = MagicMock()
            mock_resp.open.side_effect = (
                urllib.error.HTTPError(
                    "url",
                    302,
                    "Redirect blocked",
                    {},
                    None,
                )
            )
            mock_opener.return_value = mock_resp
            result = discover_services("example.com")
            assert result is None


class TestDiscoverServices:
    def setup_method(self):
        _cache.clear()

    def _mock_success(self, mock_dns, mock_opener):
        """Set up mocks for a successful discovery."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(
            SAMPLE_DISCOVERY
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_open = MagicMock()
        mock_open.open.return_value = mock_resp
        mock_opener.return_value = mock_open
        return mock_resp

    @patch(
        "autogpt.utils.agent_discovery.urllib.request"
        ".build_opener"
    )
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_success(self, mock_dns, mock_opener):
        self._mock_success(mock_dns, mock_opener)
        result = discover_services("example.com")
        assert result is not None
        assert result.domain == "example.com"
        assert result.has_service("memory")

    @patch(
        "autogpt.utils.agent_discovery.urllib.request"
        ".build_opener"
    )
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_accepts_201(self, mock_dns, mock_opener):
        """Non-200 2xx responses should succeed."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status = 201
        mock_resp.read.return_value = json.dumps(
            SAMPLE_DISCOVERY
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_open = MagicMock()
        mock_open.open.return_value = mock_resp
        mock_opener.return_value = mock_open
        result = discover_services("example.com")
        assert result is not None

    @patch(
        "autogpt.utils.agent_discovery.urllib.request"
        ".build_opener"
    )
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_404_returns_none(self, mock_dns, mock_opener):
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_open = MagicMock()
        mock_open.open.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, None
        )
        mock_opener.return_value = mock_open
        result = discover_services("example.com")
        assert result is None

    @patch(
        "autogpt.utils.agent_discovery.urllib.request"
        ".build_opener"
    )
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_404_negative_cached(self, mock_dns, mock_opener):
        """404 should be negative-cached."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_open = MagicMock()
        mock_open.open.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, None
        )
        mock_opener.return_value = mock_open
        discover_services("example.com")
        assert "example.com" in _cache
        assert _cache["example.com"][1] is None

    @patch(
        "autogpt.utils.agent_discovery.urllib.request"
        ".build_opener"
    )
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_timeout_not_cached(self, mock_dns, mock_opener):
        """Transient timeout should NOT be negative-cached."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_open = MagicMock()
        mock_open.open.side_effect = TimeoutError()
        mock_opener.return_value = mock_open
        result = discover_services("example.com")
        assert result is None
        assert "example.com" not in _cache

    @patch(
        "autogpt.utils.agent_discovery.urllib.request"
        ".build_opener"
    )
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_malformed_json_not_cached(
        self, mock_dns, mock_opener
    ):
        """Malformed JSON should NOT be negative-cached."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_open = MagicMock()
        mock_open.open.return_value = mock_resp
        mock_opener.return_value = mock_open
        result = discover_services("example.com")
        assert result is None

    @patch(
        "autogpt.utils.agent_discovery.urllib.request"
        ".build_opener"
    )
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_cache_hit(self, mock_dns, mock_opener):
        self._mock_success(mock_dns, mock_opener)
        r1 = discover_services("example.com")
        r2 = discover_services("example.com")
        assert r1 is not None
        assert r2 is not None
        # opener.open called only once (cache hit)
        opener_inst = mock_opener.return_value
        assert opener_inst.open.call_count == 1

    @patch(
        "autogpt.utils.agent_discovery.urllib.request"
        ".build_opener"
    )
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_cache_expired(self, mock_dns, mock_opener):
        self._mock_success(mock_dns, mock_opener)
        # Inject expired cache
        _cache["example.com"] = (
            time.time() - 7200,
            SAMPLE_DISCOVERY,
        )
        discover_services("example.com")
        opener_inst = mock_opener.return_value
        assert opener_inst.open.call_count == 1

    @patch(
        "autogpt.utils.agent_discovery.urllib.request"
        ".build_opener"
    )
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_pinned_ip_in_url(self, mock_dns, mock_opener):
        """URL should use pinned IP, not domain (TOCTOU)."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(
            SAMPLE_DISCOVERY
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_open = MagicMock()
        mock_open.open.return_value = mock_resp
        mock_opener.return_value = mock_open

        discover_services("example.com")

        # Check the URL used the pinned IP
        call_args = mock_open.open.call_args
        req = call_args[0][0]
        assert "93.184.216.34" in req.full_url
        assert req.get_header("Host") == "example.com"
