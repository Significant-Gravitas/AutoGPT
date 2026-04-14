"""Tests for Agent Discovery Protocol client."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from autogpt.utils.agent_discovery import (
    _MAX_ADP_BODY_BYTES,
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


def _mock_response(status=200, body=None, content_length=None):
    """Build a mock http.client.HTTPResponse.

    Always explicitly configure getheader so the bounded-read
    path doesn't try to int() a MagicMock placeholder.
    """
    if body is None:
        body = json.dumps(SAMPLE_DISCOVERY).encode()
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = body
    mock_resp.getheader.return_value = content_length
    return mock_resp


def _mock_conn(resp, raise_on_request=None):
    """Build a mock HTTPSConnection that returns the given response."""
    mock_c = MagicMock()
    mock_c.getresponse.return_value = resp
    if raise_on_request:
        mock_c.request.side_effect = raise_on_request
    return mock_c


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
        original = result.get_service("memory")
        assert original["endpoint"] != "MUTATED"

    def test_deepcopy_isolation_services_property(self):
        result = DiscoveryResult(SAMPLE_DISCOVERY)
        svcs = result.services
        svcs["memory"]["endpoint"] = "MUTATED"
        assert result.get_service("memory")["endpoint"] != "MUTATED"


class TestDiscoveryResultSchemaValidation:
    """Validate the payload shape up front."""

    def test_rejects_non_dict_payload(self):
        with pytest.raises(
            ValueError, match="JSON object"
        ):
            DiscoveryResult(["not", "a", "dict"])

    def test_rejects_string_payload(self):
        with pytest.raises(
            ValueError, match="JSON object"
        ):
            DiscoveryResult("nope")

    def test_rejects_services_not_list(self):
        with pytest.raises(
            ValueError, match="services"
        ):
            DiscoveryResult({"services": {"memory": {}}})

    def test_rejects_service_entry_not_dict(self):
        with pytest.raises(
            ValueError, match="objects"
        ):
            DiscoveryResult(
                {"services": ["just-a-string"]}
            )

    def test_rejects_service_missing_name(self):
        with pytest.raises(
            ValueError, match="name"
        ):
            DiscoveryResult(
                {
                    "services": [
                        {"endpoint": "https://x.com/y"}
                    ]
                }
            )

    def test_rejects_service_name_empty(self):
        with pytest.raises(
            ValueError, match="name"
        ):
            DiscoveryResult(
                {"services": [{"name": ""}]}
            )

    def test_rejects_service_name_not_string(self):
        with pytest.raises(
            ValueError, match="name"
        ):
            DiscoveryResult(
                {"services": [{"name": 42}]}
            )

    def test_accepts_empty_services_list(self):
        """Empty services list is valid (domain publishes
        ADP but offers nothing right now)."""
        result = DiscoveryResult(
            {
                "agent_discovery_version": "0.1",
                "domain": "example.com",
                "services": [],
            }
        )
        assert result.list_services() == []

    def test_accepts_missing_services_key(self):
        """Missing 'services' defaults to empty list."""
        result = DiscoveryResult(
            {"agent_discovery_version": "0.1"}
        )
        assert result.list_services() == []


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

    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_rejects_mixed_public_and_private(self, mock_dns):
        """If ANY resolved IP is blocked, reject the whole
        lookup. Attacker must not be able to mix one private
        entry into a multi-A record."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("10.0.0.1", 443)),
        ]
        with pytest.raises(ValueError, match="blocked address"):
            discover_services("mixed.example.com")


class TestDiscoverServices:
    def setup_method(self):
        _cache.clear()

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_success(self, mock_dns, mock_conn_cls):
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_conn_cls.return_value = _mock_conn(
            _mock_response()
        )

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
        mock_conn_cls.return_value = _mock_conn(
            _mock_response(status=201)
        )

        result = discover_services("example2.com")
        assert result is not None

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_redirect_blocked(self, mock_dns, mock_conn_cls):
        """3xx redirects return None (SSRF bypass prevention)."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_conn_cls.return_value = _mock_conn(
            _mock_response(status=302, body=b"")
        )

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
        mock_conn_cls.return_value = _mock_conn(
            _mock_response(status=404, body=b"")
        )

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
        mock_conn_cls.return_value = _mock_conn(
            _mock_response(),
            raise_on_request=TimeoutError(),
        )

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
        mock_conn_cls.return_value = _mock_conn(
            _mock_response(body=b"not json")
        )

        test_domain = "example6.com"
        result = discover_services(test_domain)
        assert result is None
        assert test_domain not in _cache

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_malformed_schema_not_cached(
        self, mock_dns, mock_conn_cls
    ):
        """Schema failure must not poison the cache."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        # Valid JSON but malformed schema
        bad_payload = json.dumps(
            {"services": [{"endpoint": "https://x.com"}]}
        ).encode()
        mock_conn_cls.return_value = _mock_conn(
            _mock_response(body=bad_payload)
        )

        test_domain = "example-bad-schema.com"
        result = discover_services(test_domain)
        assert result is None
        assert test_domain not in _cache

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_cache_hit_skips_dns(
        self, mock_dns, mock_conn_cls
    ):
        """Cache hit should NOT call DNS or HTTPSConnection."""
        _cache["example7.com"] = (
            time.time(),
            SAMPLE_DISCOVERY,
        )
        result = discover_services("example7.com")
        assert result is not None
        mock_dns.assert_not_called()
        mock_conn_cls.assert_not_called()

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_cache_hit_empty_dict_not_mistaken_for_miss(
        self, mock_dns, mock_conn_cls
    ):
        """Edge case: a cached empty-dict positive result
        must not be treated as a negative entry just
        because `{}` is falsy."""
        test_domain = "empty.example.com"
        _cache[test_domain] = (time.time(), {})
        result = discover_services(test_domain)
        # Should be a DiscoveryResult, not None
        assert result is not None
        assert isinstance(result, DiscoveryResult)
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
        mock_conn_cls.return_value = _mock_conn(
            _mock_response()
        )

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
        mock_conn_cls.return_value = _mock_conn(
            _mock_response()
        )

        discover_services("example10.com")

        call_args = mock_conn_cls.call_args
        first_arg = call_args[0][0]
        assert first_arg == "example10.com"


class TestMultiIPFailover:
    """Verify we try each resolved IP on transport failure."""

    def setup_method(self):
        _cache.clear()

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_second_ip_succeeds_after_first_fails(
        self, mock_dns, mock_conn_cls
    ):
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("93.184.216.35", 443)),
        ]
        # First connect raises, second succeeds
        good_conn = _mock_conn(_mock_response())
        bad_conn = _mock_conn(
            _mock_response(),
            raise_on_request=ConnectionRefusedError(),
        )
        mock_conn_cls.side_effect = [bad_conn, good_conn]

        result = discover_services("failover1.example.com")
        assert result is not None
        assert mock_conn_cls.call_count == 2

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_all_ips_fail_returns_none_no_cache(
        self, mock_dns, mock_conn_cls
    ):
        """All IPs exhausted: return None, don't cache."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("93.184.216.35", 443)),
        ]
        mock_conn_cls.return_value = _mock_conn(
            _mock_response(),
            raise_on_request=ConnectionRefusedError(),
        )

        test_domain = "alldown.example.com"
        result = discover_services(test_domain)
        assert result is None
        assert test_domain not in _cache
        # Each IP attempted
        assert mock_conn_cls.call_count == 2

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_404_on_first_ip_does_not_try_second(
        self, mock_dns, mock_conn_cls
    ):
        """Authoritative response: don't retry on other IPs."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("93.184.216.35", 443)),
        ]
        mock_conn_cls.return_value = _mock_conn(
            _mock_response(status=404, body=b"")
        )

        result = discover_services("authnotfound.example.com")
        assert result is None
        # Only ONE HTTPSConnection constructed
        assert mock_conn_cls.call_count == 1

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_redirect_on_first_ip_does_not_try_second(
        self, mock_dns, mock_conn_cls
    ):
        """Redirect is an authoritative response, not transport."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("93.184.216.35", 443)),
        ]
        mock_conn_cls.return_value = _mock_conn(
            _mock_response(status=302, body=b"")
        )

        result = discover_services("redir.example.com")
        assert result is None
        assert mock_conn_cls.call_count == 1

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_conn_closed_on_transport_error(
        self, mock_dns, mock_conn_cls
    ):
        """conn.close() must be called even when the request
        raises, so we don't leak sockets when every call to
        a flaky domain hits an error."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        bad = _mock_conn(
            _mock_response(),
            raise_on_request=ConnectionRefusedError(),
        )
        mock_conn_cls.return_value = bad

        discover_services("leaky.example.com")
        bad.close.assert_called_once()

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_all_failover_attempts_close_their_conn(
        self, mock_dns, mock_conn_cls
    ):
        """Each IP attempted must close its own conn."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("93.184.216.35", 443)),
            (2, 1, 6, "", ("93.184.216.36", 443)),
        ]
        conns = [
            _mock_conn(
                _mock_response(),
                raise_on_request=ConnectionRefusedError(),
            )
            for _ in range(3)
        ]
        mock_conn_cls.side_effect = conns

        discover_services("allclosed.example.com")
        for c in conns:
            c.close.assert_called_once()

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_conn_closed_on_success(
        self, mock_dns, mock_conn_cls
    ):
        """Success path also closes conn (in finally)."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        good = _mock_conn(_mock_response())
        mock_conn_cls.return_value = good

        result = discover_services("success-close.example.com")
        assert result is not None
        good.close.assert_called_once()


class TestBodySizeCap:
    """Verify the 1 MiB response body cap."""

    def setup_method(self):
        _cache.clear()

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_rejects_oversized_content_length(
        self, mock_dns, mock_conn_cls
    ):
        """Content-Length over cap -> reject fast."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_conn_cls.return_value = _mock_conn(
            _mock_response(
                content_length=str(_MAX_ADP_BODY_BYTES + 1)
            )
        )

        test_domain = "toobig.example.com"
        result = discover_services(test_domain)
        assert result is None
        # Not negative-cached (malformed, not 404)
        assert test_domain not in _cache

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_rejects_oversized_actual_body(
        self, mock_dns, mock_conn_cls
    ):
        """Body bytes over cap (chunked/no CL) -> reject."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        huge_body = b"x" * (_MAX_ADP_BODY_BYTES + 100)
        mock_conn_cls.return_value = _mock_conn(
            _mock_response(body=huge_body)
        )

        test_domain = "bigbody.example.com"
        result = discover_services(test_domain)
        assert result is None
        assert test_domain not in _cache

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_accepts_body_under_cap(
        self, mock_dns, mock_conn_cls
    ):
        """Normal-size body passes."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        normal = json.dumps(SAMPLE_DISCOVERY).encode()
        mock_conn_cls.return_value = _mock_conn(
            _mock_response(
                body=normal,
                content_length=str(len(normal)),
            )
        )

        result = discover_services("normalsize.example.com")
        assert result is not None

    @patch("http.client.HTTPSConnection")
    @patch("autogpt.utils.agent_discovery.socket.getaddrinfo")
    def test_malformed_content_length_rejected(
        self, mock_dns, mock_conn_cls
    ):
        """Garbage Content-Length -> reject (don't trust it)."""
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ]
        mock_conn_cls.return_value = _mock_conn(
            _mock_response(content_length="not-a-number")
        )

        result = discover_services("badcl.example.com")
        assert result is None
