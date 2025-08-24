"""
Comprehensive integration tests for authentication dependencies.
Tests the full authentication flow from HTTP requests to user validation.
"""

import os
from unittest import mock

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from autogpt_libs.auth.config import Settings
from autogpt_libs.auth.dependencies import (
    get_user_id,
    requires_admin_user,
    requires_user,
)
from autogpt_libs.auth.models import DEFAULT_USER_ID, User


class TestAuthDependencies:
    """Test suite for authentication dependency functions."""

    @pytest.fixture
    def app(self):
        """Create a test FastAPI application."""
        app = FastAPI()

        @app.get("/user")
        def get_user_endpoint(user: User = Depends(requires_user)):
            return {"user_id": user.user_id, "role": user.role}

        @app.get("/admin")
        def get_admin_endpoint(user: User = Depends(requires_admin_user)):
            return {"user_id": user.user_id, "role": user.role}

        @app.get("/user-id")
        def get_user_id_endpoint(user_id: str = Depends(get_user_id)):
            return {"user_id": user_id}

        return app

    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return TestClient(app)

    def test_requires_user_with_valid_jwt_payload(self):
        """Test requires_user with valid JWT payload."""
        jwt_payload = {"sub": "user-123", "role": "user", "email": "user@example.com"}

        # Mock get_jwt_payload to return our test payload
        with mock.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        ):
            user = requires_user(jwt_payload)
            assert isinstance(user, User)
            assert user.user_id == "user-123"
            assert user.role == "user"

    def test_requires_user_with_admin_jwt_payload(self):
        """Test requires_user accepts admin users."""
        jwt_payload = {
            "sub": "admin-456",
            "role": "admin",
            "email": "admin@example.com",
        }

        with mock.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        ):
            user = requires_user(jwt_payload)
            assert user.user_id == "admin-456"
            assert user.role == "admin"

    def test_requires_user_missing_sub(self):
        """Test requires_user with missing user ID."""
        jwt_payload = {"role": "user", "email": "user@example.com"}

        with pytest.raises(HTTPException) as exc_info:
            requires_user(jwt_payload)
        assert exc_info.value.status_code == 401
        assert "User ID not found" in exc_info.value.detail

    def test_requires_user_empty_sub(self):
        """Test requires_user with empty user ID."""
        jwt_payload = {"sub": "", "role": "user"}

        with pytest.raises(HTTPException) as exc_info:
            requires_user(jwt_payload)
        assert exc_info.value.status_code == 401

    def test_requires_admin_user_with_admin(self):
        """Test requires_admin_user with admin role."""
        jwt_payload = {
            "sub": "admin-789",
            "role": "admin",
            "email": "admin@example.com",
        }

        with mock.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        ):
            user = requires_admin_user(jwt_payload)
            assert user.user_id == "admin-789"
            assert user.role == "admin"

    def test_requires_admin_user_with_regular_user(self):
        """Test requires_admin_user rejects regular users."""
        jwt_payload = {"sub": "user-123", "role": "user", "email": "user@example.com"}

        with pytest.raises(HTTPException) as exc_info:
            requires_admin_user(jwt_payload)
        assert exc_info.value.status_code == 403
        assert "Admin access required" in exc_info.value.detail

    def test_requires_admin_user_missing_role(self):
        """Test requires_admin_user with missing role."""
        jwt_payload = {"sub": "user-123", "email": "user@example.com"}

        with pytest.raises(KeyError):
            requires_admin_user(jwt_payload)

    def test_get_user_id_with_valid_payload(self):
        """Test get_user_id extracts user ID correctly."""
        jwt_payload = {"sub": "user-id-xyz", "role": "user"}

        with mock.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        ):
            user_id = get_user_id(jwt_payload)
            assert user_id == "user-id-xyz"

    def test_get_user_id_missing_sub(self):
        """Test get_user_id with missing user ID."""
        jwt_payload = {"role": "user"}

        with pytest.raises(HTTPException) as exc_info:
            get_user_id(jwt_payload)
        assert exc_info.value.status_code == 401
        assert "User ID not found" in exc_info.value.detail

    def test_get_user_id_none_sub(self):
        """Test get_user_id with None user ID."""
        jwt_payload = {"sub": None, "role": "user"}

        with pytest.raises(HTTPException) as exc_info:
            get_user_id(jwt_payload)
        assert exc_info.value.status_code == 401

    def test_auth_disabled_flow(self):
        """Test authentication flow when auth is disabled."""
        with mock.patch.dict(os.environ, {"ENABLE_AUTH": "false"}, clear=True):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                # When auth is disabled, default payload is used
                jwt_payload = {"sub": DEFAULT_USER_ID, "role": "admin"}

                user = requires_user(jwt_payload)
                assert user.user_id == DEFAULT_USER_ID
                assert user.role == "admin"

                admin = requires_admin_user(jwt_payload)
                assert admin.role == "admin"

                user_id = get_user_id(jwt_payload)
                assert user_id == DEFAULT_USER_ID


class TestAuthDependenciesIntegration:
    """Integration tests for auth dependencies with FastAPI."""

    @pytest.fixture
    def create_token(self):
        """Helper to create JWT tokens."""
        import jwt

        def _create_token(payload, secret="test-secret-with-proper-length-123456"):
            return jwt.encode(payload, secret, algorithm="HS256")

        return _create_token

    def test_endpoint_auth_disabled(self):
        """Test endpoints work without auth when disabled."""
        with mock.patch.dict(os.environ, {"ENABLE_AUTH": "false"}, clear=True):
            app = FastAPI()

            @app.get("/test")
            def test_endpoint(user: User = Depends(requires_user)):
                return {"user_id": user.user_id}

            # Need to patch the settings at import time
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                client = TestClient(app)

                # No auth header needed when auth is disabled
                response = client.get("/test")
                assert response.status_code == 200
                assert response.json()["user_id"] == DEFAULT_USER_ID

    def test_endpoint_auth_enabled_no_token(self):
        """Test endpoints require token when auth is enabled."""
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_AUTH": "true",
                "SUPABASE_JWT_SECRET": "test-secret-with-proper-length-123456",
            },
            clear=True,
        ):
            app = FastAPI()

            @app.get("/test")
            def test_endpoint(user: User = Depends(requires_user)):
                return {"user_id": user.user_id}

            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                client = TestClient(app)

                # Should fail without auth header
                response = client.get("/test")
                assert response.status_code == 401

    def test_endpoint_with_valid_token(self, create_token):
        """Test endpoint with valid JWT token."""
        secret = "test-secret-with-proper-length-123456"
        with mock.patch.dict(
            os.environ,
            {"ENABLE_AUTH": "true", "SUPABASE_JWT_SECRET": secret},
            clear=True,
        ):
            app = FastAPI()

            @app.get("/test")
            def test_endpoint(user: User = Depends(requires_user)):
                return {"user_id": user.user_id, "role": user.role}

            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                client = TestClient(app)

                token = create_token(
                    {"sub": "test-user", "role": "user", "aud": "authenticated"},
                    secret=secret,
                )

                response = client.get(
                    "/test", headers={"Authorization": f"Bearer {token}"}
                )
                assert response.status_code == 200
                assert response.json()["user_id"] == "test-user"

    def test_admin_endpoint_requires_admin_role(self, create_token):
        """Test admin endpoint rejects non-admin users."""
        secret = "test-secret-with-proper-length-123456"
        with mock.patch.dict(
            os.environ,
            {"ENABLE_AUTH": "true", "SUPABASE_JWT_SECRET": secret},
            clear=True,
        ):
            app = FastAPI()

            @app.get("/admin")
            def admin_endpoint(user: User = Depends(requires_admin_user)):
                return {"user_id": user.user_id}

            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                client = TestClient(app)

                # Regular user token
                user_token = create_token(
                    {"sub": "regular-user", "role": "user", "aud": "authenticated"},
                    secret=secret,
                )

                response = client.get(
                    "/admin", headers={"Authorization": f"Bearer {user_token}"}
                )
                assert response.status_code == 403

                # Admin token
                admin_token = create_token(
                    {"sub": "admin-user", "role": "admin", "aud": "authenticated"},
                    secret=secret,
                )

                response = client.get(
                    "/admin", headers={"Authorization": f"Bearer {admin_token}"}
                )
                assert response.status_code == 200
                assert response.json()["user_id"] == "admin-user"


class TestAuthDependenciesEdgeCases:
    """Edge case tests for authentication dependencies."""

    def test_dependency_with_complex_payload(self):
        """Test dependencies handle complex JWT payloads."""
        complex_payload = {
            "sub": "user-123",
            "role": "admin",
            "email": "test@example.com",
            "app_metadata": {"provider": "email", "providers": ["email"]},
            "user_metadata": {
                "full_name": "Test User",
                "avatar_url": "https://example.com/avatar.jpg",
            },
            "aud": "authenticated",
            "iat": 1234567890,
            "exp": 9999999999,
        }

        user = requires_user(complex_payload)
        assert user.user_id == "user-123"
        assert user.email == "test@example.com"

        admin = requires_admin_user(complex_payload)
        assert admin.role == "admin"

    def test_dependency_with_unicode_in_payload(self):
        """Test dependencies handle unicode in JWT payloads."""
        unicode_payload = {
            "sub": "user-😀-123",
            "role": "user",
            "email": "测试@example.com",
            "name": "日本語",
        }

        user = requires_user(unicode_payload)
        assert "😀" in user.user_id
        assert user.email == "测试@example.com"

    def test_dependency_with_null_values(self):
        """Test dependencies handle null values in payload."""
        null_payload = {
            "sub": "user-123",
            "role": "user",
            "email": None,
            "phone": None,
            "metadata": None,
        }

        user = requires_user(null_payload)
        assert user.user_id == "user-123"
        assert user.email is None

    def test_concurrent_requests_isolation(self):
        """Test that concurrent requests don't interfere with each other."""
        payload1 = {"sub": "user-1", "role": "user"}
        payload2 = {"sub": "user-2", "role": "admin"}

        # Simulate concurrent processing
        user1 = requires_user(payload1)
        user2 = requires_admin_user(payload2)

        assert user1.user_id == "user-1"
        assert user2.user_id == "user-2"
        assert user1.role == "user"
        assert user2.role == "admin"

    def test_dependency_error_propagation(self):
        """Test that errors propagate correctly through dependencies."""
        # Enable auth for this test
        with mock.patch.dict(
            os.environ,
            {"ENABLE_AUTH": "true", "SUPABASE_JWT_SECRET": "test-secret-123456"},
            clear=True,
        ):
            from autogpt_libs.auth import jwt_utils

            with mock.patch.object(jwt_utils, "settings", Settings()):
                # Test various error conditions
                error_payloads = [
                    (None, "Authorization header is missing"),  # None payload
                    ({}, "User ID not found"),  # Missing sub
                    ({"sub": ""}, "User ID not found"),  # Empty sub
                    ({"sub": "user", "role": "user"}, None),  # Valid for requires_user
                    (
                        {"sub": "user", "role": "user"},
                        "Admin access required",
                    ),  # Invalid for requires_admin_user
                ]

                # Test None payload
                with pytest.raises(HTTPException) as exc_info:
                    requires_user(error_payloads[0][0])
                assert error_payloads[0][1] in exc_info.value.detail

                # Test empty dict and empty sub
                for payload, expected_error in error_payloads[1:3]:
                    with pytest.raises(HTTPException) as exc_info:
                        requires_user(payload)
                    assert expected_error in exc_info.value.detail

                # Valid case
                user = requires_user(error_payloads[3][0])
                assert user.user_id == "user"

                # Admin check
                with pytest.raises(HTTPException) as exc_info:
                    requires_admin_user(error_payloads[4][0])
                assert error_payloads[4][1] in exc_info.value.detail
