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
    "trust": {"verification_url": "https://example.com/verify"},
}


class TestDiscoveryResult:
    def test_list_services(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        assert sorted(result.list_services()) == ["identity", "memory"]

    def test_has_service(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        assert result.has_service("memory") is True
        assert result.has_service("nonexistent") is False

    def test_get_service(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        memory = result.get_service("memory")
        assert memory is not None
        assert memory["endpoint"] == "https://example.com/api/memory"

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


class TestSSRFValidation:
    def test_rejects_ip_literal(self):
        with pytest.raises(ValueError, match="invalid domain"):
            discover_services("127.0.0.1")

    def test_rejects_ip_with_port(self):
        with pytest.raises(ValueError, match="invalid domain"):
            discover_services("127.0.0.1:8080")

    def test_rejects_localhost(self):
        with pytest.raises(ValueError, match="blocked address"):
            discover_services("localhost")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            discover_services("")

    def test_rejects_scheme_injection(self):
        with pytest.raises(ValueError, match="invalid domain"):
            discover_services("http://evil.com")

    def test_rejects_userinfo_injection(self):
        with pytest.raises(ValueError, match="invalid domain"):
            discover_services("admin:pass@evil.com")

    @patch("socket.getaddrinfo")
    def test_rejects_private_ip_resolution(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 6, "", ("10.0.0.1", 443))
        ]
        with pytest.raises(ValueError, match="blocked address"):
            discover_services("evil.example.com")

    @patch("socket.getaddrinfo")
    def test_rejects_metadata_ip(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 6, "", ("169.254.169.254", 443))
        ]
        with pytest.raises(ValueError, match="blocked address"):
            discover_services("metadata.example.com")


class TestDiscoverServices:
    def setup_method(self):
        _cache.clear()

    @patch("urllib.request.urlopen")
    @patch("socket.getaddrinfo")
    def test_success(self, mock_dns, mock_urlopen):
        mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(
            SAMPLE_DISCOVERY
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = discover_services("example.com")
        assert result is not None
        assert result.domain == "example.com"
        assert result.has_service("memory")

    @patch("urllib.request.urlopen")
    @patch("socket.getaddrinfo")
    def test_404_returns_none(self, mock_dns, mock_urlopen):
        mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, None
        )
        result = discover_services("example.com")
        assert result is None

    @patch("urllib.request.urlopen")
    @patch("socket.getaddrinfo")
    def test_timeout_returns_none(self, mock_dns, mock_urlopen):
        mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        mock_urlopen.side_effect = TimeoutError()
        result = discover_services("example.com")
        assert result is None

    @patch("urllib.request.urlopen")
    @patch("socket.getaddrinfo")
    def test_malformed_json_returns_none(self, mock_dns, mock_urlopen):
        mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        result = discover_services("example.com")
        assert result is None

    @patch("urllib.request.urlopen")
    @patch("socket.getaddrinfo")
    def test_cache_hit(self, mock_dns, mock_urlopen):
        mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(
            SAMPLE_DISCOVERY
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result1 = discover_services("example.com")
        result2 = discover_services("example.com")
        assert result1 is not None
        assert result2 is not None
        assert mock_urlopen.call_count == 1  # Second call used cache

    @patch("urllib.request.urlopen")
    @patch("socket.getaddrinfo")
    def test_cache_miss_after_ttl(self, mock_dns, mock_urlopen):
        mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(
            SAMPLE_DISCOVERY
        ).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        # Manually inject expired cache entry
        _cache["example.com"] = (time.time() - 7200, SAMPLE_DISCOVERY)
        discover_services("example.com")
        assert mock_urlopen.call_count == 1  # Cache expired, made request

    @patch("urllib.request.urlopen")
    @patch("socket.getaddrinfo")
    def test_negative_cache(self, mock_dns, mock_urlopen):
        mock_dns.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, None
        )
        result1 = discover_services("example.com")
        result2 = discover_services("example.com")
        assert result1 is None
        assert result2 is None
        assert mock_urlopen.call_count == 1  # Negative cached
