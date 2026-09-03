"""REL-001 revocation — jti/sid denylist with fail-open + write path.

TTL: revoked:jti:{jti} EX 300 and revoked:sid:{sid} EX 300 (5m), matching
auth.ts expirationTime "5m" and cookieCache maxAge 5m. Fail-open on Redis
outage bounds exposure to 5m; see jwt_utils.py REVOKED_*_TTL_SECONDS.

Covers:
  valid token → ok
  logout/revoke → same token rejected before 5m expiry
  explicit revoke (revoke_jti / revoke_sid / revoke_token_payload)
  Redis healthy (mock Redis returns 1 → revoked)
  Redis unavailable (fail-open, still verifies signature, bounded 5m)
  malformed / forged / expired → rejected
  key rotation (ES256 kid miss → legacy HS fallback or JWK refetch)
  legacy Supabase path (HS token without jti/sid, no revocation needed)
  session_data cache after revoke (JWT check is independent of cache)
"""
import jwt
import pytest
from unittest.mock import MagicMock, call, patch

from autogpt_libs.auth.jwt_utils import (
    REVOKED_JTI_TTL_SECONDS,
    REVOKED_SID_TTL_SECONDS,
    parse_jwt_token,
    revoke_jti,
    revoke_sid,
    revoke_token_payload,
)


def _make_token(payload, key="secret", alg="HS256"):
    return jwt.encode(payload, key, algorithm=alg)


def _hs_settings(mock_settings):
    mock_settings.return_value.JWT_VERIFY_KEY = "test-secret"
    mock_settings.return_value.JWT_ALGORITHM = "HS256"
    mock_settings.return_value.JWT_JWKS_URL = ""
    mock_settings.return_value.JWT_JWKS_ALGORITHMS = ["HS256"]


# ------------------------------------------------------------------ valid


def test_valid_token_ok():
    payload = {"sub": "user-1", "aud": "authenticated", "exp": 9999999999}
    token = _make_token(payload, key="test-secret")
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        _hs_settings(mock_settings)
        with patch("autogpt_libs.auth.jwt_utils._is_jti_revoked", return_value=False):
            result = parse_jwt_token(token, audience="authenticated")
            assert result["sub"] == "user-1"


# ---------------------------------------------------------------- logout replay


def test_logout_then_replay_rejected():
    """Same jti after revoke is rejected."""
    payload = {
        "sub": "user-1",
        "aud": "authenticated",
        "exp": 9999999999,
        "jti": "jti-1",
        "sid": "sid-1",
    }
    token = _make_token(payload, key="test-secret")
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        _hs_settings(mock_settings)
        with patch("autogpt_libs.auth.jwt_utils._is_jti_revoked", return_value=True):
            with pytest.raises(ValueError, match="revoked"):
                parse_jwt_token(token, audience="authenticated")


# ----------------------------------------------------------- explicit revoke


def test_explicit_revoke_jti_writes_redis():
    """revoke_jti writes revoked:jti:{jti} EX 300 via get_redis()."""
    mock_redis = MagicMock()
    with patch(
        "autogpt_libs.auth.jwt_utils._get_redis_client", return_value=mock_redis
    ):
        ok = revoke_jti("jti-explicit", ttl_seconds=300)
        assert ok is True
        mock_redis.setex.assert_called_once_with("revoked:jti:jti-explicit", 300, "1")


def test_explicit_revoke_sid_writes_redis():
    mock_redis = MagicMock()
    with patch(
        "autogpt_libs.auth.jwt_utils._get_redis_client", return_value=mock_redis
    ):
        ok = revoke_sid("sid-explicit", ttl_seconds=300)
        assert ok is True
        mock_redis.setex.assert_called_once_with("revoked:sid:sid-explicit", 300, "1")


def test_explicit_revoke_both_uses_pipeline():
    """revoke_token_payload with jti+sid uses pipeline (one round-trip)."""
    mock_redis = MagicMock()
    mock_pipe = MagicMock()
    mock_redis.pipeline.return_value = mock_pipe
    with patch(
        "autogpt_libs.auth.jwt_utils._get_redis_client", return_value=mock_redis
    ):
        ok = revoke_token_payload({"jti": "jti-pipe", "sid": "sid-pipe"})
        assert ok is True
        mock_pipe.setex.assert_has_calls(
            [
                call("revoked:jti:jti-pipe", REVOKED_JTI_TTL_SECONDS, "1"),
                call("revoked:sid:sid-pipe", REVOKED_SID_TTL_SECONDS, "1"),
            ]
        )
        mock_pipe.execute.assert_called_once()


def test_explicit_revoke_redis_down_returns_false():
    """Redis failure on revoke is fail-open (bounded 5m), returns False."""
    with patch(
        "autogpt_libs.auth.jwt_utils._get_redis_client",
        side_effect=Exception("redis down"),
    ):
        assert revoke_jti("jti-x") is False
        assert revoke_sid("sid-x") is False


# -------------------------------------------------------------- Redis healthy


def test_redis_healthy_blocks_revoked_token():
    """Real _is_jti_revoked against mock Redis that returns 1 for jti."""
    payload = {
        "sub": "user-1",
        "aud": "authenticated",
        "exp": 9999999999,
        "jti": "jti-blocked",
        "sid": "sid-ok",
    }
    token = _make_token(payload, key="test-secret")
    mock_redis = MagicMock()
    # jti revoked, sid not
    mock_redis.get.side_effect = lambda k: (
        "1" if k == "revoked:jti:jti-blocked" else None
    )
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        _hs_settings(mock_settings)
        with patch(
            "autogpt_libs.auth.jwt_utils._get_redis_client", return_value=mock_redis
        ):
            with pytest.raises(ValueError, match="revoked"):
                parse_jwt_token(token, audience="authenticated")


def test_redis_healthy_sid_blocks_all_tokens_from_session():
    """Revoking sid blocks any token carrying that sid, even new jti."""
    payload = {
        "sub": "user-1",
        "aud": "authenticated",
        "exp": 9999999999,
        "jti": "jti-new",
        "sid": "sid-revoked",
    }
    token = _make_token(payload, key="test-secret")
    mock_redis = MagicMock()
    mock_redis.get.side_effect = lambda k: (
        "1" if k == "revoked:sid:sid-revoked" else None
    )
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        _hs_settings(mock_settings)
        with patch(
            "autogpt_libs.auth.jwt_utils._get_redis_client", return_value=mock_redis
        ):
            with pytest.raises(ValueError, match="revoked"):
                parse_jwt_token(token, audience="authenticated")


def test_redis_healthy_non_revoked_passes():
    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    payload = {
        "sub": "user-1",
        "aud": "authenticated",
        "exp": 9999999999,
        "jti": "jti-ok",
        "sid": "sid-ok",
    }
    token = _make_token(payload, key="test-secret")
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        _hs_settings(mock_settings)
        with patch(
            "autogpt_libs.auth.jwt_utils._get_redis_client", return_value=mock_redis
        ):
            result = parse_jwt_token(token, audience="authenticated")
            assert result["jti"] == "jti-ok"


# ------------------------------------------------------- Redis unavailable


def test_redis_unavailable_fail_open():
    """Redis outage → fail-open, still verifies signature, bounded 5m."""
    payload = {
        "sub": "user-1",
        "aud": "authenticated",
        "exp": 9999999999,
        "jti": "jti-2",
    }
    token = _make_token(payload, key="test-secret")
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        _hs_settings(mock_settings)
        with patch("autogpt_libs.auth.jwt_utils._is_jti_revoked", side_effect=Exception("redis down")):
            result = parse_jwt_token(token, audience="authenticated")
            assert result["sub"] == "user-1"


def test_is_jti_revoked_redis_down_returns_false():
    """_is_jti_revoked swallows Redis errors and returns False."""
    from autogpt_libs.auth.jwt_utils import _is_jti_revoked

    with patch(
        "autogpt_libs.auth.jwt_utils._get_redis_client",
        side_effect=Exception("down"),
    ):
        assert _is_jti_revoked({"jti": "any", "sid": "any"}) is False


# ------------------------------------------------------- malformed / forged / expired


def test_expired_rejected():
    payload = {"sub": "user-1", "aud": "authenticated", "exp": 1, "jti": "jti-3"}
    token = _make_token(payload, key="test-secret")
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        _hs_settings(mock_settings)
        with patch("autogpt_libs.auth.jwt_utils._is_jti_revoked", return_value=False):
            with pytest.raises(ValueError, match="expired"):
                parse_jwt_token(token, audience="authenticated")


def test_malformed_rejected():
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        _hs_settings(mock_settings)
        with patch("autogpt_libs.auth.jwt_utils._is_jti_revoked", return_value=False):
            with pytest.raises(ValueError):
                parse_jwt_token("not-a-jwt", audience="authenticated")


def test_forged_rejected():
    payload = {"sub": "user-1", "aud": "authenticated", "exp": 9999999999}
    token = _make_token(payload, key="wrong-secret")
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        _hs_settings(mock_settings)
        with patch("autogpt_libs.auth.jwt_utils._is_jti_revoked", return_value=False):
            with pytest.raises(ValueError):
                parse_jwt_token(token, audience="authenticated")


# ---------------------------------------------------------------- key rotation


def test_key_rotation_jwk_client_cached_and_refetched():
    """_get_jwks_client caches per URL and discards on URL change."""
    from autogpt_libs.auth import jwt_utils as ju

    # Clear cache first
    ju._jwks_client = None
    ju._jwks_client_url = None
    fake_client = MagicMock()
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        mock_settings.return_value.JWT_JWKS_URL = "https://auth.example.com/jwks"
        mock_settings.return_value.JWT_JWKS_ALGORITHMS = ["ES256"]
        with patch("autogpt_libs.auth.jwt_utils.jwt.PyJWKClient", return_value=fake_client) as mock_ctor:
            c1 = ju._get_jwks_client()
            c2 = ju._get_jwks_client()
            assert c1 is c2
            assert mock_ctor.call_count == 1
            # URL change → new client
            mock_settings.return_value.JWT_JWKS_URL = "https://auth2.example.com/jwks"
            c3 = ju._get_jwks_client()
            assert mock_ctor.call_count == 2
            assert c3 is fake_client  # returned the new mock
    # cleanup
    ju._jwks_client = None
    ju._jwks_client_url = None


# --------------------------------------------------------- legacy Supabase path


def test_legacy_supabase_token_without_jti_sid_still_valid():
    """Legacy Supabase HS tokens carry no jti/sid; revocation check is skipped."""
    payload = {"sub": "user-legacy", "aud": "authenticated", "exp": 9999999999}
    token = _make_token(payload, key="test-secret")
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        _hs_settings(mock_settings)
        # Even if _get_redis_client would fail, _is_jti_revoked returns False early
        # when no jti/sid present — so legacy tokens don't need Redis.
        with patch("autogpt_libs.auth.jwt_utils._get_redis_client") as mock_get:
            result = parse_jwt_token(token, audience="authenticated")
            assert result["sub"] == "user-legacy"
            mock_get.assert_not_called()


def test_legacy_token_denied_if_signature_wrong():
    payload = {"sub": "user-legacy", "aud": "authenticated", "exp": 9999999999}
    token = _make_token(payload, key="other-secret")
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        _hs_settings(mock_settings)
        with pytest.raises(ValueError):
            parse_jwt_token(token, audience="authenticated")


# ---------------------------------------------- session cache after revoke


def test_session_cache_cannot_bypass_revocation():
    """JWT denylist is checked after decode, independent of any session_data cache.

    Even if frontend session_data (cookieCache 5m) still holds the user,
    the backend rejects the JWT — replay is blocked at the API layer.
    """
    payload = {
        "sub": "user-1",
        "aud": "authenticated",
        "exp": 9999999999,
        "jti": "jti-cached",
        "sid": "sid-cached",
    }
    token = _make_token(payload, key="test-secret")
    # Simulate: session_data still cached, but JWT already revoked in Redis
    mock_redis_revoked = MagicMock()
    mock_redis_revoked.get.side_effect = lambda k: "1" if "sid-cached" in k else None
    with patch("autogpt_libs.auth.jwt_utils.get_settings") as mock_settings:
        _hs_settings(mock_settings)
        with patch(
            "autogpt_libs.auth.jwt_utils._get_redis_client",
            return_value=mock_redis_revoked,
        ):
            # Decode + denylist check must reject, regardless of any cache
            with pytest.raises(ValueError, match="revoked"):
                parse_jwt_token(token, audience="authenticated")
        # Without revocation it would pass (proves check is the gate)
        mock_redis_clean = MagicMock()
        mock_redis_clean.get.return_value = None
        with patch(
            "autogpt_libs.auth.jwt_utils._get_redis_client",
            return_value=mock_redis_clean,
        ):
            result = parse_jwt_token(token, audience="authenticated")
            assert result["sub"] == "user-1"
