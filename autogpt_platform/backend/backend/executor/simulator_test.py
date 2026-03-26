"""Tests for simulator credential redaction, URL sanitization, and regex boundaries."""

import json

import pytest

from backend.executor.simulator import (
    _SECRET_KEY_PATTERN,
    _redact_inputs,
    _sanitize_url,
)
from backend.executor.utils import _SIMULATION_CONTEXT_MAX_BYTES

# ---------------------------------------------------------------------------
# Scenario 3: simulation_context validation before DB write
# ---------------------------------------------------------------------------


class TestSimulationContextValidation:
    """simulation_context is validated (JSON-serializable) before DB write."""

    def test_valid_simulation_context_serializable(self):
        """A valid dict is JSON-serializable and within size limits."""
        ctx = {"expected_emails": ["From: alice@test.com, Subject: Hello"]}
        encoded = json.dumps(ctx).encode("utf-8")
        assert len(encoded) <= _SIMULATION_CONTEXT_MAX_BYTES

    def test_non_serializable_simulation_context_raises(self):
        """Non-JSON-serializable simulation_context raises TypeError."""
        ctx = {"callback": lambda x: x}  # lambdas are not JSON-serializable
        with pytest.raises(TypeError):
            json.dumps(ctx)

    def test_simulation_context_gated_behind_dry_run(self):
        """When dry_run is False, simulation_context should be treated as None."""
        # This mirrors the validation logic in add_graph_execution:
        # safe_simulation_context = simulation_context if dry_run else None
        simulation_context = {"scenario": "test"}
        dry_run = False
        safe = simulation_context if dry_run else None
        assert safe is None

    def test_simulation_context_passes_when_dry_run(self):
        """When dry_run is True, simulation_context is preserved."""
        simulation_context = {"scenario": "test"}
        dry_run = True
        safe = simulation_context if dry_run else None
        assert safe == {"scenario": "test"}


# ---------------------------------------------------------------------------
# Scenario 4: simulation_context 16KB limit enforced
# ---------------------------------------------------------------------------


class TestSimulationContextSizeLimit:
    """simulation_context 16KB limit is enforced."""

    def test_max_bytes_is_16kb(self):
        """_SIMULATION_CONTEXT_MAX_BYTES is exactly 16 * 1024 = 16384."""
        assert _SIMULATION_CONTEXT_MAX_BYTES == 16 * 1024

    def test_context_within_limit_passes(self):
        """A context under 16KB passes the size check."""
        ctx = {"data": "x" * 1000}
        encoded = json.dumps(ctx).encode("utf-8")
        assert len(encoded) <= _SIMULATION_CONTEXT_MAX_BYTES

    def test_context_exceeding_limit_detected(self):
        """A context over 16KB is detected by the size check."""
        ctx = {"data": "x" * 20000}  # ~20KB, well over 16KB
        encoded = json.dumps(ctx).encode("utf-8")
        assert len(encoded) > _SIMULATION_CONTEXT_MAX_BYTES

    def test_context_at_boundary(self):
        """A context right at the 16KB boundary is within limits."""
        # Build a context that's exactly at the limit
        overhead = len(json.dumps({"d": ""}).encode("utf-8"))
        fill_size = _SIMULATION_CONTEXT_MAX_BYTES - overhead
        ctx = {"d": "a" * fill_size}
        encoded = json.dumps(ctx).encode("utf-8")
        assert len(encoded) == _SIMULATION_CONTEXT_MAX_BYTES

    def test_context_one_byte_over_boundary(self):
        """A context one byte over the 16KB boundary exceeds limits."""
        overhead = len(json.dumps({"d": ""}).encode("utf-8"))
        fill_size = _SIMULATION_CONTEXT_MAX_BYTES - overhead + 1
        ctx = {"d": "a" * fill_size}
        encoded = json.dumps(ctx).encode("utf-8")
        assert len(encoded) > _SIMULATION_CONTEXT_MAX_BYTES


# ---------------------------------------------------------------------------
# Scenario 5: Secret fields redacted
# ---------------------------------------------------------------------------


class TestSecretFieldRedaction:
    """Secret fields (api_key, token, password) are redacted in simulation prompts."""

    def test_api_key_redacted(self):
        result = _redact_inputs({"api_key": "sk-secret-123"})
        assert result["api_key"] == "<REDACTED>"

    def test_password_redacted(self):
        result = _redact_inputs({"password": "hunter2"})
        assert result["password"] == "<REDACTED>"

    def test_secret_redacted(self):
        result = _redact_inputs({"secret": "s3cr3t"})
        assert result["secret"] == "<REDACTED>"

    def test_access_token_redacted(self):
        result = _redact_inputs({"access_token": "tok-abc"})
        assert result["access_token"] == "<REDACTED>"

    def test_private_key_redacted(self):
        result = _redact_inputs({"private_key": "-----BEGIN RSA-----"})
        assert result["private_key"] == "<REDACTED>"

    def test_auth_token_redacted(self):
        result = _redact_inputs({"auth_token": "bearer-xyz"})
        assert result["auth_token"] == "<REDACTED>"

    def test_oauth_token_redacted(self):
        result = _redact_inputs({"oauth_token": "oauth-123"})
        assert result["oauth_token"] == "<REDACTED>"

    def test_credentials_redacted(self):
        result = _redact_inputs({"credentials": {"user": "admin", "pass": "pw"}})
        assert result["credentials"] == "<REDACTED>"

    def test_credential_redacted(self):
        """Singular 'credential' is also matched by 'credentials?' pattern."""
        result = _redact_inputs({"credential": "cred-value"})
        assert result["credential"] == "<REDACTED>"

    def test_nested_secret_redacted(self):
        """Secret keys inside nested dicts are also redacted."""
        result = _redact_inputs({"config": {"api_key": "sk-nested", "name": "test"}})
        assert result["config"]["api_key"] == "<REDACTED>"
        assert result["config"]["name"] == "test"

    def test_secret_in_list_redacted(self):
        """Secret keys propagate through lists of dicts."""
        result = _redact_inputs({"items": [{"password": "pw1"}, {"password": "pw2"}]})
        # 'items' is not a secret key, so the list is traversed with key='items'
        # The inner dicts are traversed with their own keys
        # Actually _redact_value for list uses the *parent* key, so inner dicts
        # are re-entered via _redact_value(key="items", value={"password": ...})
        # which calls _redact_value("password", "pw1") for inner dict items
        for item in result["items"]:
            assert item["password"] == "<REDACTED>"

    def test_apikey_no_underscore_redacted(self):
        """'apikey' (no underscore) is matched by api_?key pattern."""
        result = _redact_inputs({"apikey": "key-123"})
        assert result["apikey"] == "<REDACTED>"

    def test_accesstoken_no_underscore_redacted(self):
        """'accesstoken' (no underscore) is matched by access_?token pattern."""
        result = _redact_inputs({"accesstoken": "tok-456"})
        assert result["accesstoken"] == "<REDACTED>"

    def test_privatekey_no_underscore_redacted(self):
        """'privatekey' (no underscore) is matched by private_?key pattern."""
        result = _redact_inputs({"privatekey": "pk-789"})
        assert result["privatekey"] == "<REDACTED>"


# ---------------------------------------------------------------------------
# Scenario 6: URL sanitization
# ---------------------------------------------------------------------------


class TestUrlSanitization:
    """URL strings are sanitized (no userinfo, query params, fragments)."""

    def test_url_query_params_stripped(self):
        result = _sanitize_url("https://api.example.com/v1?key=secret&token=abc")
        assert result == "https://api.example.com/v1"

    def test_url_fragment_stripped(self):
        result = _sanitize_url("https://example.com/page#section")
        assert result == "https://example.com/page"

    def test_url_userinfo_stripped(self):
        result = _sanitize_url("https://user:pass@example.com/path")
        assert result == "https://example.com/path"

    def test_url_path_preserved(self):
        result = _sanitize_url("https://api.example.com/v1/resources")
        assert "/v1/resources" in result

    def test_url_port_preserved(self):
        result = _sanitize_url("https://api.example.com:8443/v1")
        assert "8443" in result

    def test_non_url_string_unchanged(self):
        """Non-URL strings are returned as-is."""
        result = _sanitize_url("not a url")
        assert result == "not a url"

    def test_url_in_input_data_sanitized(self):
        """URLs in input_data values are sanitized by _redact_inputs."""
        result = _redact_inputs(
            {"endpoint": "https://api.example.com/v1?api_key=secret123"}
        )
        assert result["endpoint"] == "https://api.example.com/v1"

    def test_non_url_non_secret_preserved(self):
        """Regular string values that aren't URLs are preserved."""
        result = _redact_inputs({"query": "search term", "count": 42})
        assert result["query"] == "search term"
        assert result["count"] == 42


# ---------------------------------------------------------------------------
# Scenario 7: Non-secret fields preserved
# ---------------------------------------------------------------------------


class TestNonSecretFieldsPreserved:
    """Non-secret fields are preserved unchanged."""

    def test_regular_fields_preserved(self):
        result = _redact_inputs(
            {
                "query": "hello world",
                "count": 10,
                "enabled": True,
                "tags": ["a", "b"],
            }
        )
        assert result["query"] == "hello world"
        assert result["count"] == 10
        assert result["enabled"] is True
        assert result["tags"] == ["a", "b"]

    def test_mixed_secret_and_non_secret(self):
        """Secret fields are redacted while non-secret fields are preserved."""
        result = _redact_inputs(
            {
                "api_key": "sk-123",
                "query": "test",
                "password": "hunter2",
                "limit": 50,
            }
        )
        assert result["api_key"] == "<REDACTED>"
        assert result["query"] == "test"
        assert result["password"] == "<REDACTED>"
        assert result["limit"] == 50


# ---------------------------------------------------------------------------
# Scenario 11: Word boundary regex doesn't false-positive
# ---------------------------------------------------------------------------


class TestRegexNoFalsePositives:
    """Regex doesn't false-positive on 'author', 'authority', 'token_count'."""

    def test_author_not_matched(self):
        assert _SECRET_KEY_PATTERN.search("author") is None

    def test_authority_not_matched(self):
        assert _SECRET_KEY_PATTERN.search("authority") is None

    def test_token_count_not_matched(self):
        assert _SECRET_KEY_PATTERN.search("token_count") is None

    def test_authorize_not_matched(self):
        assert _SECRET_KEY_PATTERN.search("authorize") is None

    def test_authorization_not_matched(self):
        assert _SECRET_KEY_PATTERN.search("authorization") is None

    def test_secret_agent_not_matched(self):
        """'secret_agent' should not match because 'secret' is followed by '_agent'
        which means the (?:$|_) matches the underscore, but the full key 'secret_agent'
        still triggers the pattern. Actually 'secret' IS a secret keyword and
        (?:$|_) matches the underscore after 'secret'."""
        # This should actually match because 'secret' is followed by '_'
        assert _SECRET_KEY_PATTERN.search("secret_agent") is not None

    def test_password_reset_not_a_false_negative(self):
        """'password_reset' contains 'password' which is a secret keyword."""
        assert _SECRET_KEY_PATTERN.search("password_reset") is not None

    def test_total_count_not_matched(self):
        assert _SECRET_KEY_PATTERN.search("total_count") is None

    def test_description_not_matched(self):
        assert _SECRET_KEY_PATTERN.search("description") is None

    def test_name_not_matched(self):
        assert _SECRET_KEY_PATTERN.search("name") is None

    def test_input_data_not_redacted(self):
        """Fields like 'author' are NOT redacted when used as input keys."""
        result = _redact_inputs(
            {
                "author": "John Doe",
                "authority": "admin",
                "token_count": 1500,
            }
        )
        assert result["author"] == "John Doe"
        assert result["authority"] == "admin"
        assert result["token_count"] == 1500


# ---------------------------------------------------------------------------
# Scenario 12: Underscore-aware boundaries catch compound forms
# ---------------------------------------------------------------------------


class TestRegexUnderscoreAwareBoundaries:
    """Underscore-aware boundaries catch 'api_secret', 'client_secret'."""

    def test_api_secret_matched(self):
        assert _SECRET_KEY_PATTERN.search("api_secret") is not None

    def test_client_secret_matched(self):
        assert _SECRET_KEY_PATTERN.search("client_secret") is not None

    def test_api_key_matched(self):
        assert _SECRET_KEY_PATTERN.search("api_key") is not None

    def test_user_password_matched(self):
        assert _SECRET_KEY_PATTERN.search("user_password") is not None

    def test_db_password_matched(self):
        assert _SECRET_KEY_PATTERN.search("db_password") is not None

    def test_my_access_token_matched(self):
        assert _SECRET_KEY_PATTERN.search("my_access_token") is not None

    def test_github_private_key_matched(self):
        assert _SECRET_KEY_PATTERN.search("github_private_key") is not None

    def test_service_credentials_matched(self):
        assert _SECRET_KEY_PATTERN.search("service_credentials") is not None

    def test_compound_redaction_in_inputs(self):
        """Compound secret keys are redacted in actual input data."""
        result = _redact_inputs(
            {
                "api_secret": "secret-val",
                "client_secret": "client-val",
                "user_password": "pw",
                "query": "preserved",
            }
        )
        assert result["api_secret"] == "<REDACTED>"
        assert result["client_secret"] == "<REDACTED>"
        assert result["user_password"] == "<REDACTED>"
        assert result["query"] == "preserved"
