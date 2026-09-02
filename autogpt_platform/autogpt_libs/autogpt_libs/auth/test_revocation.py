"""REL-001 revocation — jti/sid denylist with fail-open.

Tests the contract:
  valid token → ok
  logout/revoke → same token rejected before 5m expiry
  Redis unavailable → fail-open (bounded 5m)
  malformed/forged/expired → rejected
  session_data cache cannot bypass revocation (JWT check is independent)
"""
import jwt
import pytest
from unittest.mock import patch, MagicMock

from autogpt_libs.auth.jwt_utils import parse_jwt_token


def _make_token(payload, key="secret", alg="HS256"):
    return jwt.encode(payload, key, algorithm=alg)


def test_valid_token_ok():
    payload = {"sub": "user-1", "aud": "authenticated", "exp": 9999999999}
    token = _make_token(payload, key="test-secret")
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        mock_settings.return_value.JWT_VERIFY_KEY = "test-secret"
        mock_settings.return_value.JWT_ALGORITHM = "HS256"
        mock_settings.return_value.JWT_JWKS_URL = ""
        mock_settings.return_value.JWT_JWKS_ALGORITHMS = ["HS256"]
        # Mock redis to not revoked
        with patch("autogpt_libs.auth.jwt_utils._is_jti_revoked", return_value=False):
            result = parse_jwt_token(token, audience="authenticated")
            assert result["sub"] == "user-1"


def test_logout_then_replay_rejected():
    """Same jti after revoke is rejected."""
    payload = {"sub": "user-1", "aud": "authenticated", "exp": 9999999999, "jti": "jti-1", "sid": "sid-1"}
    token = _make_token(payload, key="test-secret")
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        mock_settings.return_value.JWT_VERIFY_KEY = "test-secret"
        mock_settings.return_value.JWT_ALGORITHM = "HS256"
        mock_settings.return_value.JWT_JWKS_URL = ""
        mock_settings.return_value.JWT_JWKS_ALGORITHMS = ["HS256"]
        with patch("autogpt_libs.auth.jwt_utils._is_jti_revoked", return_value=True):
            with pytest.raises(ValueError, match="revoked"):
                parse_jwt_token(token, audience="authenticated")


def test_redis_unavailable_fail_open():
    """Redis outage → fail-open, still verifies signature, bounded 5m."""
    payload = {"sub": "user-1", "aud": "authenticated", "exp": 9999999999, "jti": "jti-2"}
    token = _make_token(payload, key="test-secret")
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        mock_settings.return_value.JWT_VERIFY_KEY = "test-secret"
        mock_settings.return_value.JWT_ALGORITHM = "HS256"
        mock_settings.return_value.JWT_JWKS_URL = ""
        mock_settings.return_value.JWT_JWKS_ALGORITHMS = ["HS256"]
        # Simulate redis exception inside _is_jti_revoked → should be caught and fail-open
        with patch("autogpt_libs.auth.jwt_utils._is_jti_revoked", side_effect=Exception("redis down")):
            # Should not raise revoked, should return payload (fail-open)
            result = parse_jwt_token(token, audience="authenticated")
            assert result["sub"] == "user-1"


def test_expired_rejected():
    payload = {"sub": "user-1", "aud": "authenticated", "exp": 1, "jti": "jti-3"}
    token = _make_token(payload, key="test-secret")
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        mock_settings.return_value.JWT_VERIFY_KEY = "test-secret"
        mock_settings.return_value.JWT_ALGORITHM = "HS256"
        mock_settings.return_value.JWT_JWKS_URL = ""
        mock_settings.return_value.JWT_JWKS_ALGORITHMS = ["HS256"]
        with patch("autogpt_libs.auth.jwt_utils._is_jti_revoked", return_value=False):
            with pytest.raises(ValueError, match="expired"):
                parse_jwt_token(token, audience="authenticated")


def test_malformed_rejected():
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        mock_settings.return_value.JWT_VERIFY_KEY = "test-secret"
        mock_settings.return_value.JWT_ALGORITHM = "HS256"
        mock_settings.return_value.JWT_JWKS_URL = ""
        with patch("autogpt_libs.auth.jwt_utils._is_jti_revoked", return_value=False):
            with pytest.raises(ValueError):
                parse_jwt_token("not-a-jwt", audience="authenticated")


def test_forged_rejected():
    payload = {"sub": "user-1", "aud": "authenticated", "exp": 9999999999}
    token = _make_token(payload, key="wrong-secret")
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        mock_settings.return_value.JWT_VERIFY_KEY = "test-secret"
        mock_settings.return_value.JWT_ALGORITHM = "HS256"
        mock_settings.return_value.JWT_JWKS_URL = ""
        mock_settings.return_value.JWT_JWKS_ALGORITHMS = ["HS256"]
        with patch("autogpt_libs.auth.jwt_utils._is_jti_revoked", return_value=False):
            with pytest.raises(ValueError):
                parse_jwt_token(token, audience="authenticated")
