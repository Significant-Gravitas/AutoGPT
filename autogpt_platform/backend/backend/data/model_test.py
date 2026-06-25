import pytest
from pydantic import SecretStr

from backend.data.model import HostScopedCredentials, NodeExecutionStats


class TestHostScopedCredentials:
    def test_host_scoped_credentials_creation(self):
        """Test creating HostScopedCredentials with required fields."""
        creds = HostScopedCredentials(
            provider="custom",
            host="api.example.com",
            headers={
                "Authorization": SecretStr("Bearer secret-token"),
                "X-API-Key": SecretStr("api-key-123"),
            },
            title="Example API Credentials",
        )

        assert creds.type == "host_scoped"
        assert creds.provider == "custom"
        assert creds.host == "api.example.com"
        assert creds.title == "Example API Credentials"
        assert len(creds.headers) == 2
        assert "Authorization" in creds.headers
        assert "X-API-Key" in creds.headers

    def test_get_headers_dict(self):
        """Test getting headers with secret values extracted."""
        creds = HostScopedCredentials(
            provider="custom",
            host="api.example.com",
            headers={
                "Authorization": SecretStr("Bearer secret-token"),
                "X-Custom-Header": SecretStr("custom-value"),
            },
        )

        headers_dict = creds.get_headers_dict()

        assert headers_dict == {
            "Authorization": "Bearer secret-token",
            "X-Custom-Header": "custom-value",
        }

    def test_matches_url_exact_host(self):
        """Test URL matching with exact host match."""
        creds = HostScopedCredentials(
            provider="custom",
            host="api.example.com",
            headers={"Authorization": SecretStr("Bearer token")},
        )

        assert creds.matches_url("https://api.example.com/v1/data")
        assert creds.matches_url("http://api.example.com/endpoint")
        assert not creds.matches_url("https://other.example.com/v1/data")
        assert not creds.matches_url("https://subdomain.api.example.com/v1/data")

    def test_matches_url_wildcard_subdomain(self):
        """Test URL matching with wildcard subdomain pattern."""
        creds = HostScopedCredentials(
            provider="custom",
            host="*.example.com",
            headers={"Authorization": SecretStr("Bearer token")},
        )

        assert creds.matches_url("https://api.example.com/v1/data")
        assert creds.matches_url("https://subdomain.example.com/endpoint")
        assert creds.matches_url("https://deep.nested.example.com/path")
        assert creds.matches_url("https://example.com/path")  # Base domain should match
        assert not creds.matches_url("https://example.org/v1/data")
        assert not creds.matches_url("https://notexample.com/v1/data")

    def test_matches_url_with_port_and_path(self):
        """Test URL matching with ports and paths."""
        creds = HostScopedCredentials(
            provider="custom",
            host="localhost",
            headers={"Authorization": SecretStr("Bearer token")},
        )

        # Non-standard ports require explicit port in credential host
        assert not creds.matches_url("http://localhost:8080/api/v1")
        assert creds.matches_url("https://localhost:443/secure/endpoint")
        assert creds.matches_url("http://localhost/simple")

    def test_matches_url_with_explicit_port(self):
        """Test URL matching with explicit port in credential host."""
        creds = HostScopedCredentials(
            provider="custom",
            host="localhost:8080",
            headers={"Authorization": SecretStr("Bearer token")},
        )

        assert creds.matches_url("http://localhost:8080/api/v1")
        assert not creds.matches_url("http://localhost:3000/api/v1")
        assert not creds.matches_url("http://localhost/simple")

    def test_empty_headers_dict(self):
        """Test HostScopedCredentials with empty headers."""
        creds = HostScopedCredentials(
            provider="custom", host="api.example.com", headers={}
        )

        assert creds.get_headers_dict() == {}
        assert creds.matches_url("https://api.example.com/test")

    def test_credential_serialization(self):
        """Test that credentials can be serialized/deserialized properly."""
        original_creds = HostScopedCredentials(
            provider="custom",
            host="api.example.com",
            headers={
                "Authorization": SecretStr("Bearer secret-token"),
                "X-API-Key": SecretStr("api-key-123"),
            },
            title="Test Credentials",
        )

        # Serialize to dict (simulating storage)
        serialized = original_creds.model_dump()

        # Deserialize back
        restored_creds = HostScopedCredentials.model_validate(serialized)

        assert restored_creds.id == original_creds.id
        assert restored_creds.provider == original_creds.provider
        assert restored_creds.host == original_creds.host
        assert restored_creds.title == original_creds.title
        assert restored_creds.type == "host_scoped"

        # Check that headers are properly restored
        assert restored_creds.get_headers_dict() == original_creds.get_headers_dict()

    @pytest.mark.parametrize(
        "host,test_url,expected",
        [
            ("api.example.com", "https://api.example.com/test", True),
            ("api.example.com", "https://different.example.com/test", False),
            ("*.example.com", "https://api.example.com/test", True),
            ("*.example.com", "https://sub.api.example.com/test", True),
            ("*.example.com", "https://example.com/test", True),
            ("*.example.com", "https://example.org/test", False),
            # Non-standard ports require explicit port in credential host
            ("localhost", "http://localhost:3000/test", False),
            ("localhost:3000", "http://localhost:3000/test", True),
            ("localhost", "http://127.0.0.1:3000/test", False),
            # IPv6 addresses (frontend stores with brackets via URL.hostname)
            ("[::1]", "http://[::1]/test", True),
            ("[::1]", "http://[::1]:80/test", True),
            ("[::1]", "https://[::1]:443/test", True),
            ("[::1]", "http://[::1]:8080/test", False),  # Non-standard port
            ("[::1]:8080", "http://[::1]:8080/test", True),
            ("[::1]:8080", "http://[::1]:9090/test", False),
            ("[2001:db8::1]", "http://[2001:db8::1]/path", True),
            ("[2001:db8::1]", "https://[2001:db8::1]:443/path", True),
            ("[2001:db8::1]", "http://[2001:db8::ff]/path", False),
        ],
    )
    def test_url_matching_parametrized(self, host: str, test_url: str, expected: bool):
        """Parametrized test for various URL matching scenarios."""
        creds = HostScopedCredentials(
            provider="test",
            host=host,
            headers={"Authorization": SecretStr("Bearer token")},
        )

        assert creds.matches_url(test_url) == expected


class TestNodeExecutionStatsIadd:
    def test_adds_numeric_fields(self):
        a = NodeExecutionStats(input_token_count=100, output_token_count=50)
        b = NodeExecutionStats(input_token_count=200, output_token_count=30)
        a += b
        assert a.input_token_count == 300
        assert a.output_token_count == 80

    def test_none_does_not_overwrite(self):
        a = NodeExecutionStats(provider_cost=0.5, error="some error")
        b = NodeExecutionStats(provider_cost=None, error=None)
        a += b
        assert a.provider_cost == 0.5
        assert a.error == "some error"

    def test_none_is_skipped_preserving_existing_value(self):
        a = NodeExecutionStats(input_token_count=100)
        b = NodeExecutionStats()
        a += b
        assert a.input_token_count == 100

    def test_dict_fields_are_merged(self):
        a = NodeExecutionStats(
            cleared_inputs={"field1": ["val1"]},
        )
        b = NodeExecutionStats(
            cleared_inputs={"field2": ["val2"]},
        )
        a += b
        assert a.cleared_inputs == {"field1": ["val1"], "field2": ["val2"]}

    def test_returns_self(self):
        a = NodeExecutionStats()
        b = NodeExecutionStats(input_token_count=10)
        result = a.__iadd__(b)
        assert result is a

    def test_not_implemented_for_non_stats(self):
        a = NodeExecutionStats()
        result = a.__iadd__("not a stats")  # type: ignore[arg-type]
        assert result is NotImplemented

    def test_error_none_does_not_clear_existing_error(self):
        a = NodeExecutionStats(error="existing error")
        b = NodeExecutionStats(error=None)
        a += b
        assert a.error == "existing error"

    def test_provider_cost_none_does_not_clear_existing_cost(self):
        a = NodeExecutionStats(provider_cost=0.05)
        b = NodeExecutionStats(provider_cost=None)
        a += b
        assert a.provider_cost == 0.05

    def test_provider_cost_accumulates_when_both_set(self):
        a = NodeExecutionStats(provider_cost=0.01)
        b = NodeExecutionStats(provider_cost=0.02)
        a += b
        assert abs((a.provider_cost or 0) - 0.03) < 1e-9

    def test_provider_cost_first_write_from_none(self):
        a = NodeExecutionStats()
        b = NodeExecutionStats(provider_cost=0.05)
        a += b
        assert a.provider_cost == 0.05

    def test_provider_cost_type_first_write_from_none(self):
        """Writing provider_cost_type into a stats with None sets it."""
        a = NodeExecutionStats()
        b = NodeExecutionStats(provider_cost_type="characters")
        a += b
        assert a.provider_cost_type == "characters"

    def test_provider_cost_type_none_does_not_overwrite(self):
        """A None provider_cost_type from other must not clear an existing value."""
        a = NodeExecutionStats(provider_cost_type="tokens")
        b = NodeExecutionStats()
        a += b
        assert a.provider_cost_type == "tokens"
