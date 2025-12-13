"""
Comprehensive tests for JWT token parsing and validation.
Ensures 100% line and branch coverage for JWT security functions.
"""

import os
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pytest_mock import MockerFixture

from autogpt_libs.auth import config, jwt_utils
from autogpt_libs.auth.config import Settings
from autogpt_libs.auth.models import User

MOCK_JWT_SECRET = "test-secret-key-with-at-least-32-characters"
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


@pytest.fixture(autouse=True)
def mock_config(mocker: MockerFixture):
    mocker.patch.dict(os.environ, {"JWT_VERIFY_KEY": MOCK_JWT_SECRET}, clear=True)
    mocker.patch.object(config, "_settings", Settings())
    yield


def create_token(payload, secret=None, algorithm="HS256"):
    """Helper to create JWT tokens."""
    if secret is None:
        secret = MOCK_JWT_SECRET
    return jwt.encode(payload, secret, algorithm=algorithm)


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
    """Test parsing a token with invalid signature."""
    # Create token with different secret
    token = create_token(TEST_USER_PAYLOAD, secret="wrong-secret")

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
    """Test verifying admin when role field is missing."""
    no_role_payload = {"sub": "user-id"}
    with pytest.raises(KeyError):
        # This will raise KeyError when checking payload["role"]
        jwt_utils.verify_user(no_role_payload, admin_only=True)


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


def test_jwt_with_different_algorithms():
    """Test that only HS256 algorithm is accepted."""
    payload = {
        "sub": "user-id",
        "role": "user",
        "aud": "authenticated",
    }

    # Try different algorithms
    algorithms = ["HS384", "HS512", "none"]
    for algo in algorithms:
        if algo == "none":
            # Special case for 'none' algorithm (security vulnerability if accepted)
            token = create_token(payload, "", algorithm="none")
        else:
            token = create_token(payload, algorithm=algo)

        with pytest.raises(ValueError) as exc_info:
            jwt_utils.parse_jwt_token(token)
        assert "Invalid token" in str(exc_info.value)
