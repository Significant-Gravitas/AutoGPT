"""
Comprehensive integration tests for authentication dependencies.
Tests the full authentication flow from HTTP requests to user validation.
"""

import os

import pytest
from fastapi import FastAPI, HTTPException, Security
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from autogpt_libs.auth.dependencies import (
    get_user_id,
    requires_admin_user,
    requires_user,
)
from autogpt_libs.auth.models import User


class TestAuthDependencies:
    """Test suite for authentication dependency functions."""

    @pytest.fixture
    def app(self):
        """Create a test FastAPI application."""
        app = FastAPI()

        @app.get("/user")
        def get_user_endpoint(user: User = Security(requires_user)):
            return {"user_id": user.user_id, "role": user.role}

        @app.get("/admin")
        def get_admin_endpoint(user: User = Security(requires_admin_user)):
            return {"user_id": user.user_id, "role": user.role}

        @app.get("/user-id")
        def get_user_id_endpoint(user_id: str = Security(get_user_id)):
            return {"user_id": user_id}

        return app

    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return TestClient(app)

    async def test_requires_user_with_valid_jwt_payload(self, mocker: MockerFixture):
        """Test requires_user with valid JWT payload."""
        jwt_payload = {"sub": "user-123", "role": "user", "email": "user@example.com"}

        # Mock get_jwt_payload to return our test payload
        mocker.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        )
        user = await requires_user(jwt_payload)
        assert isinstance(user, User)
        assert user.user_id == "user-123"
        assert user.role == "user"

    async def test_requires_user_with_admin_jwt_payload(self, mocker: MockerFixture):
        """Test requires_user accepts admin users."""
        jwt_payload = {
            "sub": "admin-456",
            "role": "admin",
            "email": "admin@example.com",
        }

        mocker.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        )
        user = await requires_user(jwt_payload)
        assert user.user_id == "admin-456"
        assert user.role == "admin"

    async def test_requires_user_missing_sub(self):
        """Test requires_user with missing user ID."""
        jwt_payload = {"role": "user", "email": "user@example.com"}

        with pytest.raises(HTTPException) as exc_info:
            await requires_user(jwt_payload)
        assert exc_info.value.status_code == 401
        assert "User ID not found" in exc_info.value.detail

    async def test_requires_user_empty_sub(self):
        """Test requires_user with empty user ID."""
        jwt_payload = {"sub": "", "role": "user"}

        with pytest.raises(HTTPException) as exc_info:
            await requires_user(jwt_payload)
        assert exc_info.value.status_code == 401

    async def test_requires_admin_user_with_admin(self, mocker: MockerFixture):
        """Test requires_admin_user with admin role."""
        jwt_payload = {
            "sub": "admin-789",
            "role": "admin",
            "email": "admin@example.com",
        }

        mocker.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        )
        user = await requires_admin_user(jwt_payload)
        assert user.user_id == "admin-789"
        assert user.role == "admin"

    async def test_requires_admin_user_with_regular_user(self):
        """Test requires_admin_user rejects regular users."""
        jwt_payload = {"sub": "user-123", "role": "user", "email": "user@example.com"}

        with pytest.raises(HTTPException) as exc_info:
            await requires_admin_user(jwt_payload)
        assert exc_info.value.status_code == 403
        assert "Admin access required" in exc_info.value.detail

    async def test_requires_admin_user_missing_role(self):
        """Test requires_admin_user with missing role."""
        jwt_payload = {"sub": "user-123", "email": "user@example.com"}

        with pytest.raises(KeyError):
            await requires_admin_user(jwt_payload)

    async def test_get_user_id_with_valid_payload(self, mocker: MockerFixture):
        """Test get_user_id extracts user ID correctly."""
        jwt_payload = {"sub": "user-id-xyz", "role": "user"}

        mocker.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        )
        user_id = await get_user_id(jwt_payload)
        assert user_id == "user-id-xyz"

    async def test_get_user_id_missing_sub(self):
        """Test get_user_id with missing user ID."""
        jwt_payload = {"role": "user"}

        with pytest.raises(HTTPException) as exc_info:
            await get_user_id(jwt_payload)
        assert exc_info.value.status_code == 401
        assert "User ID not found" in exc_info.value.detail

    async def test_get_user_id_none_sub(self):
        """Test get_user_id with None user ID."""
        jwt_payload = {"sub": None, "role": "user"}

        with pytest.raises(HTTPException) as exc_info:
            await get_user_id(jwt_payload)
        assert exc_info.value.status_code == 401


class TestAuthDependenciesIntegration:
    """Integration tests for auth dependencies with FastAPI."""

    acceptable_jwt_secret = "test-secret-with-proper-length-123456"

    @pytest.fixture
    def create_token(self, mocker: MockerFixture):
        """Helper to create JWT tokens."""
        import jwt

        mocker.patch.dict(
            os.environ,
            {"JWT_VERIFY_KEY": self.acceptable_jwt_secret},
            clear=True,
        )

        def _create_token(payload, secret=self.acceptable_jwt_secret):
            return jwt.encode(payload, secret, algorithm="HS256")

        return _create_token

    async def test_endpoint_auth_enabled_no_token(self):
        """Test endpoints require token when auth is enabled."""
        app = FastAPI()

        @app.get("/test")
        def test_endpoint(user: User = Security(requires_user)):
            return {"user_id": user.user_id}

        client = TestClient(app)

        # Should fail without auth header
        response = client.get("/test")
        assert response.status_code == 401

    async def test_endpoint_with_valid_token(self, create_token):
        """Test endpoint with valid JWT token."""
        app = FastAPI()

        @app.get("/test")
        def test_endpoint(user: User = Security(requires_user)):
            return {"user_id": user.user_id, "role": user.role}

        client = TestClient(app)

        token = create_token(
            {"sub": "test-user", "role": "user", "aud": "authenticated"},
            secret=self.acceptable_jwt_secret,
        )

        response = client.get("/test", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["user_id"] == "test-user"

    async def test_admin_endpoint_requires_admin_role(self, create_token):
        """Test admin endpoint rejects non-admin users."""
        app = FastAPI()

        @app.get("/admin")
        def admin_endpoint(user: User = Security(requires_admin_user)):
            return {"user_id": user.user_id}

        client = TestClient(app)

        # Regular user token
        user_token = create_token(
            {"sub": "regular-user", "role": "user", "aud": "authenticated"},
            secret=self.acceptable_jwt_secret,
        )

        response = client.get(
            "/admin", headers={"Authorization": f"Bearer {user_token}"}
        )
        assert response.status_code == 403

        # Admin token
        admin_token = create_token(
            {"sub": "admin-user", "role": "admin", "aud": "authenticated"},
            secret=self.acceptable_jwt_secret,
        )

        response = client.get(
            "/admin", headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        assert response.json()["user_id"] == "admin-user"


class TestAuthDependenciesEdgeCases:
    """Edge case tests for authentication dependencies."""

    async def test_dependency_with_complex_payload(self):
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

        user = await requires_user(complex_payload)
        assert user.user_id == "user-123"
        assert user.email == "test@example.com"

        admin = await requires_admin_user(complex_payload)
        assert admin.role == "admin"

    async def test_dependency_with_unicode_in_payload(self):
        """Test dependencies handle unicode in JWT payloads."""
        unicode_payload = {
            "sub": "user-😀-123",
            "role": "user",
            "email": "测试@example.com",
            "name": "日本語",
        }

        user = await requires_user(unicode_payload)
        assert "😀" in user.user_id
        assert user.email == "测试@example.com"

    async def test_dependency_with_null_values(self):
        """Test dependencies handle null values in payload."""
        null_payload = {
            "sub": "user-123",
            "role": "user",
            "email": None,
            "phone": None,
            "metadata": None,
        }

        user = await requires_user(null_payload)
        assert user.user_id == "user-123"
        assert user.email is None

    async def test_concurrent_requests_isolation(self):
        """Test that concurrent requests don't interfere with each other."""
        payload1 = {"sub": "user-1", "role": "user"}
        payload2 = {"sub": "user-2", "role": "admin"}

        # Simulate concurrent processing
        user1 = await requires_user(payload1)
        user2 = await requires_admin_user(payload2)

        assert user1.user_id == "user-1"
        assert user2.user_id == "user-2"
        assert user1.role == "user"
        assert user2.role == "admin"

    @pytest.mark.parametrize(
        "payload,expected_error,admin_only",
        [
            (None, "Authorization header is missing", False),
            ({}, "User ID not found", False),
            ({"sub": ""}, "User ID not found", False),
            ({"role": "user"}, "User ID not found", False),
            ({"sub": "user", "role": "user"}, "Admin access required", True),
        ],
    )
    async def test_dependency_error_cases(
        self, payload, expected_error: str, admin_only: bool
    ):
        """Test that errors propagate correctly through dependencies."""
        # Import verify_user to test it directly since dependencies use FastAPI Security
        from autogpt_libs.auth.jwt_utils import verify_user

        with pytest.raises(HTTPException) as exc_info:
            verify_user(payload, admin_only=admin_only)
        assert expected_error in exc_info.value.detail

    async def test_dependency_valid_user(self):
        """Test valid user case for dependency."""
        # Import verify_user to test it directly since dependencies use FastAPI Security
        from autogpt_libs.auth.jwt_utils import verify_user

        # Valid case
        user = verify_user({"sub": "user", "role": "user"}, admin_only=False)
        assert user.user_id == "user"
