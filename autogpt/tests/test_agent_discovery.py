"""Tests for Agent Discovery Protocol client."""

import json
import time
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


def _make_mock_https_conn(
    status=200,
    body=None,
    raise_on_request=None,
):
    """Build a mock HTTPSConnection class."""
    if body is None:
        body = json.dumps(SAMPLE_DISCOVERY).encode()

    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = body

    mock_conn = MagicMock()
    mock_conn.getresponse.return_value = mock_resp
    if raise_on_request:
        mock_conn.request.side_effect = raise_on_request

    # Mock the constructor
    mock_conn_cls = MagicMock(return_value=mock_conn)
    return mock_conn_cls, mock_conn


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
        assert "memory" in memory["endpoint"]

    def test_get_service_missing(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        assert result.get_service("nonexistent") is None

    def test_domain_property(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        assert result.domain == SAMPLE_DISCOVERY["domain"]

    def test_version_property(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        assert result.version == "0.1"

    def test_trust_property(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        assert "verification_url" in result.trust

    def test_deepcopy_isolation_get_service(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        svc = result.get_service("memory")
        svc["endpoint"] = "MUTATED"
        # Original should be unchanged
        original = result.get_service("memory")
        assert original["endpoint"] != "MUTATED"

    def test_deepcopy_isolation_services_property(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        svcs = result.services
        svcs["memory"]["endpoint"] = "MUTATED"
        assert result.get_service("memory")["endpoint"] != "MUTATED"


class TestSSRFValidation:
    def setup_method(self):
        _cache.clear()

    def test_rejects_ip_literal(self):
        with pytest.raises(ValueError, match="invalid domain"):
            discover_services("127.0.0.1")

    def test_rejects_ip_with_port(self):
        with pytest.raises(ValueError, match="invalid domain"):
            discover_services("127.0.0.1:8080")

    def test_rejects_localhost_format(self):
        """localhost has no TLD, fails FQDN regex."""
        with pytest.raises(ValueError, match="invalid domain"):
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

    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_rejects_private_ip_resolution(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 6, "", ("10.0.0.1", 443))
        ]
        with pytest.raises(ValueError, match="blocked address"):
            discover_services("evil1.example.com")

    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_rejects_metadata_ip(self, mock_dns):
        mock_dns.return_value = [
            (2, 1, 6, "", ("169.254.169.254", 443))
        ]
        with pytest.raises(ValueError, match="blocked address"):
            discover_services("evil2.example.com")

    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_rejects_loopback_v6(self, mock_dns):
        mock_dns.return_value = [
            (10, 1, 6, "", ("::1", 443, 0, 0))
        ]
        with pytest.raises(ValueError, match="blocked address"):
            discover_services("evil3.example.com")


class TestDiscoverServices:
    def setup_method(self):
        _cache.clear()

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_success(self, mock_dns, mock_conn_cls):
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(
            SAMPLE_DISCOVERY
        ).encode()
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp
        mock_conn_cls.return_value = mock_conn

        result = discover_services("example1.com")
        assert result is not None
        assert result.domain == "example.com"
        assert result.has_service("memory")

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_accepts_201(self, mock_dns, mock_conn_cls):
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status = 201
        mock_resp.read.return_value = json.dumps(
            SAMPLE_DISCOVERY
        ).encode()
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp
        mock_conn_cls.return_value = mock_conn

        result = discover_services("example2.com")
        assert result is not None

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_redirect_blocked(self, mock_dns, mock_conn_cls):
        """3xx redirects return None (SSRF bypass prevention)."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status = 302
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp
        mock_conn_cls.return_value = mock_conn

        result = discover_services("example3.com")
        assert result is None

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_404_returns_none_and_caches(
        self, mock_dns, mock_conn_cls
    ):
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status = 404
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp
        mock_conn_cls.return_value = mock_conn

        test_domain = "example4.com"
        result = discover_services(test_domain)
        assert result is None
        assert test_domain in _cache
        assert _cache[test_domain][1] is None

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_timeout_not_cached(self, mock_dns, mock_conn_cls):
        """Transient timeout should NOT be negative-cached."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_conn = MagicMock()
        mock_conn.request.side_effect = TimeoutError()
        mock_conn_cls.return_value = mock_conn

        test_domain = "example5.com"
        result = discover_services(test_domain)
        assert result is None
        assert test_domain not in _cache

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_malformed_json_not_cached(
        self, mock_dns, mock_conn_cls
    ):
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"not json"
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp
        mock_conn_cls.return_value = mock_conn

        test_domain = "example6.com"
        result = discover_services(test_domain)
        assert result is None
        assert test_domain not in _cache

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_cache_hit_skips_dns(
        self, mock_dns, mock_conn_cls
    ):
        """Cache hit should NOT call DNS or HTTPSConnection."""
        # Populate cache
        _cache["example7.com"] = (
            time.time(),
            SAMPLE_DISCOVERY,
        )
        result = discover_services("example7.com")
        assert result is not None
        # Cache check happens BEFORE DNS, so these should
        # never be called
        mock_dns.assert_not_called()
        mock_conn_cls.assert_not_called()

    @patch("http.client.HTTPSConnection")
    @patch(
        "autogpt.utils.agent_discovery._resolve_and_validate"
    )
    def test_cache_hit_survives_dns_failure(
        self, mock_resolve, mock_conn_cls
    ):
        """Cached result returned even if DNS broken."""
        _cache["example8.com"] = (
            time.time(),
            SAMPLE_DISCOVERY,
        )
        mock_resolve.side_effect = ValueError("DNS broken")
        # Should NOT raise — cache hit happens first
        result = discover_services("example8.com")
        assert result is not None
        mock_resolve.assert_not_called()

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_cache_expired_triggers_fetch(
        self, mock_dns, mock_conn_cls
    ):
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(
            SAMPLE_DISCOVERY
        ).encode()
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp
        mock_conn_cls.return_value = mock_conn

        # Expired cache
        _cache["example9.com"] = (
            time.time() - 7200,
            SAMPLE_DISCOVERY,
        )
        discover_services("example9.com")
        mock_conn_cls.assert_called_once()

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_https_connection_uses_domain_for_sni(
        self, mock_dns, mock_conn_cls
    ):
        """HTTPSConnection should be constructed with domain
        so SSL SNI/cert validation uses domain name."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(
            SAMPLE_DISCOVERY
        ).encode()
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp
        mock_conn_cls.return_value = mock_conn

        discover_services("example10.com")

        # First positional arg to HTTPSConnection
        # should be the domain, not the IP
        call_args = mock_conn_cls.call_args
        first_arg = call_args[0][0]
        assert first_arg == "example10.com"
