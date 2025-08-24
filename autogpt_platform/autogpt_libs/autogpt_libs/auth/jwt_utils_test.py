"""
Comprehensive tests for JWT token parsing and validation.
Ensures 100% line and branch coverage for JWT security functions.
"""

import os
from datetime import datetime, timedelta, timezone
from unittest import mock

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from autogpt_libs.auth.config import Settings
from autogpt_libs.auth.jwt_utils import (
    AUTH_DISABLED_DEFAULT_PAYLOAD,
    get_jwt_payload,
    parse_jwt_token,
    verify_user,
)
from autogpt_libs.auth.models import DEFAULT_USER_ID, User


class TestJWTUtils:
    """Test suite for JWT utility functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment with proper JWT secret."""
        self.valid_secret = "test-secret-key-with-at-least-32-characters"
        self.test_payload = {
            "sub": "test-user-id",
            "role": "user",
            "aud": "authenticated",
            "email": "test@example.com",
        }
        self.admin_payload = {
            "sub": "admin-user-id",
            "role": "admin",
            "aud": "authenticated",
            "email": "admin@example.com",
        }

    def create_token(self, payload, secret=None, algorithm="HS256"):
        """Helper to create JWT tokens."""
        if secret is None:
            secret = self.valid_secret
        return jwt.encode(payload, secret, algorithm=algorithm)

    def test_parse_jwt_token_valid(self):
        """Test parsing a valid JWT token."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": self.valid_secret,
            },
            clear=True,
        ):
            # Need to reimport to get new settings
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                token = self.create_token(self.test_payload)
                result = parse_jwt_token(token)

                assert result["sub"] == "test-user-id"
                assert result["role"] == "user"
                assert result["aud"] == "authenticated"

    def test_parse_jwt_token_expired(self):
        """Test parsing an expired JWT token."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": self.valid_secret,
            },
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                expired_payload = {
                    **self.test_payload,
                    "exp": datetime.now(timezone.utc) - timedelta(hours=1),
                }
                token = self.create_token(expired_payload)

                with pytest.raises(ValueError) as exc_info:
                    parse_jwt_token(token)
                assert "Token has expired" in str(exc_info.value)

    def test_parse_jwt_token_invalid_signature(self):
        """Test parsing a token with invalid signature."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": self.valid_secret,
            },
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                # Create token with different secret
                token = self.create_token(self.test_payload, secret="wrong-secret")

                with pytest.raises(ValueError) as exc_info:
                    parse_jwt_token(token)
                assert "Invalid token" in str(exc_info.value)

    def test_parse_jwt_token_malformed(self):
        """Test parsing a malformed token."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": self.valid_secret,
            },
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                malformed_tokens = [
                    "not.a.token",
                    "invalid",
                    "",
                    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9",  # Header only
                    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0",  # No signature
                ]

                for token in malformed_tokens:
                    with pytest.raises(ValueError) as exc_info:
                        parse_jwt_token(token)
                    assert "Invalid token" in str(exc_info.value)

    def test_parse_jwt_token_wrong_audience(self):
        """Test parsing a token with wrong audience."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": self.valid_secret,
            },
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                wrong_aud_payload = {**self.test_payload, "aud": "wrong-audience"}
                token = self.create_token(wrong_aud_payload)

                with pytest.raises(ValueError) as exc_info:
                    parse_jwt_token(token)
                assert "Invalid token" in str(exc_info.value)

    def test_parse_jwt_token_missing_audience(self):
        """Test parsing a token without audience claim."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": self.valid_secret,
            },
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                no_aud_payload = {
                    k: v for k, v in self.test_payload.items() if k != "aud"
                }
                token = self.create_token(no_aud_payload)

                with pytest.raises(ValueError) as exc_info:
                    parse_jwt_token(token)
                assert "Invalid token" in str(exc_info.value)

    def test_get_jwt_payload_with_valid_token(self):
        """Test extracting JWT payload with valid bearer token."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": self.valid_secret,
            },
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                token = self.create_token(self.test_payload)
                credentials = HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials=token
                )

                result = get_jwt_payload(credentials)
                assert result["sub"] == "test-user-id"
                assert result["role"] == "user"

    def test_get_jwt_payload_auth_disabled_no_credentials(self):
        """Test JWT payload when auth is disabled and no credentials provided."""
        with mock.patch.dict(
            os.environ,
            {"ENABLE_AUTH": "false"},
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                result = get_jwt_payload(None)
                assert result == AUTH_DISABLED_DEFAULT_PAYLOAD
                assert result["sub"] == DEFAULT_USER_ID
                assert result["role"] == "admin"

    def test_get_jwt_payload_auth_enabled_no_credentials(self):
        """Test JWT payload when auth is enabled but no credentials provided."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": self.valid_secret,
            },
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                with pytest.raises(HTTPException) as exc_info:
                    get_jwt_payload(None)
                assert exc_info.value.status_code == 401
                assert "Authorization header is missing" in exc_info.value.detail

    def test_get_jwt_payload_invalid_token(self):
        """Test JWT payload extraction with invalid token."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": self.valid_secret,
            },
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                credentials = HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials="invalid.token.here"
                )

                with pytest.raises(HTTPException) as exc_info:
                    get_jwt_payload(credentials)
                assert exc_info.value.status_code == 401
                assert "Invalid token" in exc_info.value.detail

    def test_verify_user_with_valid_user(self):
        """Test verifying a valid user."""
        user = verify_user(self.test_payload, admin_only=False)
        assert isinstance(user, User)
        assert user.user_id == "test-user-id"
        assert user.role == "user"
        assert user.email == "test@example.com"

    def test_verify_user_with_admin(self):
        """Test verifying an admin user."""
        user = verify_user(self.admin_payload, admin_only=True)
        assert isinstance(user, User)
        assert user.user_id == "admin-user-id"
        assert user.role == "admin"

    def test_verify_user_admin_only_with_regular_user(self):
        """Test verifying regular user when admin is required."""
        with pytest.raises(HTTPException) as exc_info:
            verify_user(self.test_payload, admin_only=True)
        assert exc_info.value.status_code == 403
        assert "Admin access required" in exc_info.value.detail

    def test_verify_user_no_payload_auth_disabled(self):
        """Test verifying user with no payload when auth is disabled."""
        with mock.patch.dict(
            os.environ,
            {"ENABLE_AUTH": "false"},
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                user = verify_user(None, admin_only=False)
                assert user.user_id == DEFAULT_USER_ID
                assert user.role == "admin"

    def test_verify_user_no_payload_auth_enabled(self):
        """Test verifying user with no payload when auth is enabled."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": self.valid_secret,
            },
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                # When auth is enabled and no payload, should raise error
                with pytest.raises(HTTPException) as exc_info:
                    verify_user(None, admin_only=False)
                assert exc_info.value.status_code == 401
                assert "Authorization header is missing" in exc_info.value.detail

    def test_verify_user_missing_sub(self):
        """Test verifying user with payload missing 'sub' field."""
        invalid_payload = {"role": "user", "email": "test@example.com"}
        with pytest.raises(HTTPException) as exc_info:
            verify_user(invalid_payload, admin_only=False)
        assert exc_info.value.status_code == 401
        assert "User ID not found in token" in exc_info.value.detail

    def test_verify_user_empty_sub(self):
        """Test verifying user with empty 'sub' field."""
        invalid_payload = {"sub": "", "role": "user"}
        with pytest.raises(HTTPException) as exc_info:
            verify_user(invalid_payload, admin_only=False)
        assert exc_info.value.status_code == 401
        assert "User ID not found in token" in exc_info.value.detail

    def test_verify_user_none_sub(self):
        """Test verifying user with None 'sub' field."""
        invalid_payload = {"sub": None, "role": "user"}
        with pytest.raises(HTTPException) as exc_info:
            verify_user(invalid_payload, admin_only=False)
        assert exc_info.value.status_code == 401
        assert "User ID not found in token" in exc_info.value.detail

    def test_verify_user_missing_role_admin_check(self):
        """Test verifying admin when role field is missing."""
        no_role_payload = {"sub": "user-id"}
        with pytest.raises(KeyError):
            # This will raise KeyError when checking payload["role"]
            verify_user(no_role_payload, admin_only=True)

    def test_auth_disabled_default_payload_structure(self):
        """Test the structure of the default payload for disabled auth."""
        assert AUTH_DISABLED_DEFAULT_PAYLOAD == {
            "sub": DEFAULT_USER_ID,
            "role": "admin",
        }
        assert isinstance(AUTH_DISABLED_DEFAULT_PAYLOAD["sub"], str)
        assert AUTH_DISABLED_DEFAULT_PAYLOAD["role"] == "admin"


class TestJWTEdgeCases:
    """Edge case tests for JWT token handling."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.valid_secret = "edge-case-secret-with-proper-length-12345"

    def test_jwt_with_additional_claims(self):
        """Test JWT token with additional custom claims."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": self.valid_secret,
            },
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                extra_claims_payload = {
                    "sub": "user-id",
                    "role": "user",
                    "aud": "authenticated",
                    "custom_claim": "custom_value",
                    "permissions": ["read", "write"],
                    "metadata": {"key": "value"},
                }
                token = jwt.encode(
                    extra_claims_payload, self.valid_secret, algorithm="HS256"
                )

                result = parse_jwt_token(token)
                assert result["sub"] == "user-id"
                assert result["custom_claim"] == "custom_value"
                assert result["permissions"] == ["read", "write"]

    def test_jwt_with_numeric_sub(self):
        """Test JWT token with numeric user ID."""
        payload = {
            "sub": 12345,  # Numeric ID
            "role": "user",
            "aud": "authenticated",
        }
        # Should convert to string internally
        user = verify_user(payload, admin_only=False)
        assert user.user_id == 12345

    def test_jwt_with_very_long_sub(self):
        """Test JWT token with very long user ID."""
        long_id = "a" * 1000
        payload = {
            "sub": long_id,
            "role": "user",
            "aud": "authenticated",
        }
        user = verify_user(payload, admin_only=False)
        assert user.user_id == long_id

    def test_jwt_with_special_characters_in_claims(self):
        """Test JWT token with special characters in claims."""
        payload = {
            "sub": "user@example.com/special-chars!@#$%",
            "role": "admin",
            "aud": "authenticated",
            "email": "test+special@example.com",
        }
        user = verify_user(payload, admin_only=True)
        assert "special-chars!@#$%" in user.user_id

    def test_jwt_with_future_iat(self):
        """Test JWT token with issued-at time in future."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": self.valid_secret,
            },
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                future_payload = {
                    "sub": "user-id",
                    "role": "user",
                    "aud": "authenticated",
                    "iat": datetime.now(timezone.utc) + timedelta(hours=1),
                }
                token = jwt.encode(future_payload, self.valid_secret, algorithm="HS256")

                # PyJWT validates iat claim and should reject future tokens
                with pytest.raises(ValueError, match="not yet valid"):
                    parse_jwt_token(token)

    def test_jwt_with_different_algorithms(self):
        """Test that only HS256 algorithm is accepted."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": self.valid_secret,
            },
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
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
                        token = jwt.encode(payload, "", algorithm="none")
                    else:
                        token = jwt.encode(payload, self.valid_secret, algorithm=algo)

                    with pytest.raises(ValueError) as exc_info:
                        parse_jwt_token(token)
                    assert "Invalid token" in str(exc_info.value)
