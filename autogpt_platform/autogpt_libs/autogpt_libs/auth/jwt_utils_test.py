"""
Comprehensive tests for JWT token parsing and validation.
Ensures 100% line and branch coverage for JWT security functions.

Verification is JWKS-only: the platform auth service issues asymmetric
(ES256) tokens, and symmetric (HS*) tokens are rejected outright.
"""

import os
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jwt.algorithms import ECAlgorithm
from pytest_mock import MockerFixture

from autogpt_libs.auth import config, jwt_utils
from autogpt_libs.auth.config import Settings
from autogpt_libs.auth.models import User

MOCK_JWKS_URL = "http://localhost:3000/api/auth/jwks"
_SIGNING_KID = "test-key-1"
TEST_USER_PAYLOAD = {
    "sub": "test-user-id",
    "role": "user",
    "aud": "authenticated",
    "email": "test@example.com",
}
TEST_ADMIN_PAYLOAD = {
    "sub": "admin-user-id",
    "role": "admin",
    "aud": "authenticated",
    "email": "admin@example.com",
}

# The active JWKS signing key, installed by the autouse fixture so the
# module-level create_token() helper can sign tokens the mocked JWK set trusts.
_signing_key: ec.EllipticCurvePrivateKey | None = None


def make_es256_keypair(kid: str = _SIGNING_KID):
    """Generate an EC P-256 keypair and the matching JWK set document."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    jwk = ECAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    jwk.update({"kid": kid, "alg": "ES256", "use": "sig"})
    return private_key, {"keys": [jwk]}


@pytest.fixture(autouse=True)
def mock_config(mocker: MockerFixture):
    """Configure a JWKS endpoint and serve a single ES256 signing key."""
    global _signing_key
    private_key, jwk_set = make_es256_keypair()
    _signing_key = private_key

    mocker.patch.dict(os.environ, {"JWT_JWKS_URL": MOCK_JWKS_URL}, clear=True)
    mocker.patch.object(config, "_settings", Settings())
    mocker.patch.object(jwt_utils, "_jwks_client", None)
    mocker.patch.object(jwt_utils, "_jwks_client_url", None)
    mocker.patch.object(jwt.PyJWKClient, "fetch_data", return_value=jwk_set)
    yield
    _signing_key = None


def create_token(payload, private_key=None, kid: str = _SIGNING_KID) -> str:
    """Helper to create ES256 JWT tokens signed by the active JWKS key."""
    if private_key is None:
        private_key = _signing_key
    return jwt.encode(payload, private_key, algorithm="ES256", headers={"kid": kid})


def test_parse_jwt_token_valid():
    """Test parsing a valid JWT token."""
    token = create_token(TEST_USER_PAYLOAD)
    result = jwt_utils.parse_jwt_token(token)

    assert result["sub"] == "test-user-id"
    assert result["role"] == "user"
    assert result["aud"] == "authenticated"


def test_parse_jwt_token_expired():
    """Test parsing an expired JWT token."""
    expired_payload = {
        **TEST_USER_PAYLOAD,
        "exp": datetime.now(timezone.utc) - timedelta(hours=1),
    }
    token = create_token(expired_payload)

    with pytest.raises(ValueError) as exc_info:
        jwt_utils.parse_jwt_token(token)
    assert "Token has expired" in str(exc_info.value)


def test_parse_jwt_token_invalid_signature():
    """A token signed by a key that is not in the JWK set is rejected."""
    other_key, _ = make_es256_keypair(kid=_SIGNING_KID)
    token = create_token(TEST_USER_PAYLOAD, private_key=other_key)

    with pytest.raises(ValueError) as exc_info:
        jwt_utils.parse_jwt_token(token)
    assert "Invalid token" in str(exc_info.value)


def test_parse_jwt_token_malformed():
    """Test parsing a malformed token."""
    malformed_tokens = [
        "not.a.token",
        "invalid",
        "",
        # Header only
        "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9",
        # No signature
        "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0",
    ]

    for token in malformed_tokens:
        with pytest.raises(ValueError) as exc_info:
            jwt_utils.parse_jwt_token(token)
        assert "Invalid token" in str(exc_info.value)


def test_parse_jwt_token_wrong_audience():
    """Test parsing a token with wrong audience."""
    wrong_aud_payload = {**TEST_USER_PAYLOAD, "aud": "wrong-audience"}
    token = create_token(wrong_aud_payload)

    with pytest.raises(ValueError) as exc_info:
        jwt_utils.parse_jwt_token(token)
    assert "Invalid token" in str(exc_info.value)


def test_parse_jwt_token_missing_audience():
    """Test parsing a token without audience claim."""
    no_aud_payload = {k: v for k, v in TEST_USER_PAYLOAD.items() if k != "aud"}
    token = create_token(no_aud_payload)

    with pytest.raises(ValueError) as exc_info:
        jwt_utils.parse_jwt_token(token)
    assert "Invalid token" in str(exc_info.value)


async def test_get_jwt_payload_with_valid_token():
    """Test extracting JWT payload with valid bearer token."""
    token = create_token(TEST_USER_PAYLOAD)
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    result = await jwt_utils.get_jwt_payload(credentials)
    assert result["sub"] == "test-user-id"
    assert result["role"] == "user"


async def test_get_jwt_payload_no_credentials():
    """Test JWT payload when no credentials provided."""
    with pytest.raises(HTTPException) as exc_info:
        await jwt_utils.get_jwt_payload(None)
    assert exc_info.value.status_code == 401
    assert "Authorization header is missing" in exc_info.value.detail


async def test_get_jwt_payload_invalid_token():
    """Test JWT payload extraction with invalid token."""
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="invalid.token.here"
    )

    with pytest.raises(HTTPException) as exc_info:
        await jwt_utils.get_jwt_payload(credentials)
    assert exc_info.value.status_code == 401
    assert "Invalid token" in exc_info.value.detail


def test_verify_user_with_valid_user():
    """Test verifying a valid user."""
    user = jwt_utils.verify_user(TEST_USER_PAYLOAD, admin_only=False)
    assert isinstance(user, User)
    assert user.user_id == "test-user-id"
    assert user.role == "user"
    assert user.email == "test@example.com"


def test_verify_user_with_admin():
    """Test verifying an admin user."""
    user = jwt_utils.verify_user(TEST_ADMIN_PAYLOAD, admin_only=True)
    assert isinstance(user, User)
    assert user.user_id == "admin-user-id"
    assert user.role == "admin"


def test_verify_user_admin_only_with_regular_user():
    """Test verifying regular user when admin is required."""
    with pytest.raises(HTTPException) as exc_info:
        jwt_utils.verify_user(TEST_USER_PAYLOAD, admin_only=True)
    assert exc_info.value.status_code == 403
    assert "Admin access required" in exc_info.value.detail


def test_verify_user_no_payload():
    """Test verifying user with no payload."""
    with pytest.raises(HTTPException) as exc_info:
        jwt_utils.verify_user(None, admin_only=False)
    assert exc_info.value.status_code == 401
    assert "Authorization header is missing" in exc_info.value.detail


def test_verify_user_missing_sub():
    """Test verifying user with payload missing 'sub' field."""
    invalid_payload = {"role": "user", "email": "test@example.com"}
    with pytest.raises(HTTPException) as exc_info:
        jwt_utils.verify_user(invalid_payload, admin_only=False)
    assert exc_info.value.status_code == 401
    assert "User ID not found in token" in exc_info.value.detail


def test_verify_user_empty_sub():
    """Test verifying user with empty 'sub' field."""
    invalid_payload = {"sub": "", "role": "user"}
    with pytest.raises(HTTPException) as exc_info:
        jwt_utils.verify_user(invalid_payload, admin_only=False)
    assert exc_info.value.status_code == 401
    assert "User ID not found in token" in exc_info.value.detail


def test_verify_user_none_sub():
    """Test verifying user with None 'sub' field."""
    invalid_payload = {"sub": None, "role": "user"}
    with pytest.raises(HTTPException) as exc_info:
        jwt_utils.verify_user(invalid_payload, admin_only=False)
    assert exc_info.value.status_code == 401
    assert "User ID not found in token" in exc_info.value.detail


def test_verify_user_missing_role_admin_check():
    """A missing 'role' claim under admin_only must fail closed (403), not
    KeyError -> 500."""
    no_role_payload = {"sub": "user-id"}
    with pytest.raises(HTTPException) as exc_info:
        jwt_utils.verify_user(no_role_payload, admin_only=True)
    assert exc_info.value.status_code == 403
    assert "Admin access required" in exc_info.value.detail


# ======================== EDGE CASES ======================== #


def test_jwt_with_additional_claims():
    """Test JWT token with additional custom claims."""
    extra_claims_payload = {
        "sub": "user-id",
        "role": "user",
        "aud": "authenticated",
        "custom_claim": "custom_value",
        "permissions": ["read", "write"],
        "metadata": {"key": "value"},
    }
    token = create_token(extra_claims_payload)

    result = jwt_utils.parse_jwt_token(token)
    assert result["sub"] == "user-id"
    assert result["custom_claim"] == "custom_value"
    assert result["permissions"] == ["read", "write"]


def test_jwt_with_numeric_sub():
    """Test JWT token with numeric user ID."""
    payload = {
        "sub": 12345,  # Numeric ID
        "role": "user",
        "aud": "authenticated",
    }
    # Should convert to string internally
    user = jwt_utils.verify_user(payload, admin_only=False)
    assert user.user_id == 12345


def test_jwt_with_very_long_sub():
    """Test JWT token with very long user ID."""
    long_id = "a" * 1000
    payload = {
        "sub": long_id,
        "role": "user",
        "aud": "authenticated",
    }
    user = jwt_utils.verify_user(payload, admin_only=False)
    assert user.user_id == long_id


def test_jwt_with_special_characters_in_claims():
    """Test JWT token with special characters in claims."""
    payload = {
        "sub": "user@example.com/special-chars!@#$%",
        "role": "admin",
        "aud": "authenticated",
        "email": "test+special@example.com",
    }
    user = jwt_utils.verify_user(payload, admin_only=True)
    assert "special-chars!@#$%" in user.user_id


def test_jwt_with_future_iat():
    """Test JWT token with issued-at time in future."""
    future_payload = {
        "sub": "user-id",
        "role": "user",
        "aud": "authenticated",
        "iat": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    token = create_token(future_payload)

    # PyJWT validates iat claim and should reject future tokens
    with pytest.raises(ValueError, match="not yet valid"):
        jwt_utils.parse_jwt_token(token)


# =============== SYMMETRIC (HS*) TOKENS ARE REJECTED =============== #


@pytest.mark.parametrize("algorithm", ["HS256", "HS384", "HS512"])
def test_parse_jwt_token_symmetric_rejected(algorithm: str):
    """Symmetric tokens are refused outright — this is the SECRT-2612 fix.

    A shared secret, if leaked (or shipped as a publicly-known default),
    would let anyone forge tokens, so the HS* path is not verified at all.
    """
    payload = {"sub": "user-id", "role": "admin", "aud": "authenticated"}
    secret = "shared-secret-at-least-64-bytes-long-for-hs512-hmac-key-padding!!"
    token = jwt.encode(payload, secret, algorithm=algorithm)

    with pytest.raises(ValueError, match="symmetric tokens are not accepted"):
        jwt_utils.parse_jwt_token(token)


def test_parse_jwt_token_none_algorithm_rejected():
    """The unsigned 'none' algorithm must never verify."""
    payload = {"sub": "user-id", "role": "admin", "aud": "authenticated"}
    token = jwt.encode(payload, "", algorithm="none")

    with pytest.raises(ValueError, match="Invalid token"):
        jwt_utils.parse_jwt_token(token)


# ==================== JWKS (ASYMMETRIC) VERIFICATION ==================== #


def test_parse_jwt_token_unknown_kid_rejected():
    """A token whose kid is absent from the JWK set is rejected."""
    other_key, _ = make_es256_keypair(kid="unknown-key")
    token = create_token(TEST_USER_PAYLOAD, private_key=other_key, kid="unknown-key")

    with pytest.raises(ValueError, match="Invalid token"):
        jwt_utils.parse_jwt_token(token)


def test_jwks_client_rekeys_when_url_changes(mocker: MockerFixture):
    """A changed JWT_JWKS_URL must produce a new client, not reuse the old
    one pointed at the previous endpoint."""
    mocker.patch.object(jwt_utils, "_jwks_client", None)
    mocker.patch.object(jwt_utils, "_jwks_client_url", None)

    mocker.patch.dict(os.environ, {"JWT_JWKS_URL": "http://first/jwks"}, clear=True)
    mocker.patch.object(config, "_settings", Settings())
    first = jwt_utils._get_jwks_client()
    assert jwt_utils._get_jwks_client() is first  # same URL -> cached

    mocker.patch.dict(os.environ, {"JWT_JWKS_URL": "http://second/jwks"}, clear=True)
    mocker.patch.object(config, "_settings", Settings())
    second = jwt_utils._get_jwks_client()

    assert second is not first
    assert second.uri == "http://second/jwks"


def test_verify_user_missing_role_is_not_a_server_error():
    """A token with no `role` claim must resolve as a plain user, not KeyError.

    verify_user already fails closed for admin_only; this covers the ordinary
    path, where indexing payload["role"] would surface as a 500 instead of a
    normal authenticated request.
    """
    no_role_payload = {"sub": "user-id", "email": "user@example.com"}

    user = jwt_utils.verify_user(no_role_payload, admin_only=False)

    assert user.user_id == "user-id"
    assert user.role == "user"
