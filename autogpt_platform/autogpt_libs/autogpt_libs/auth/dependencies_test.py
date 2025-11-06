"""
Comprehensive integration tests for authentication dependencies.
Tests the full authentication flow from HTTP requests to user validation.
"""

import os
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException, Request, Security
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_requires_user_missing_sub(self):
        """Test requires_user with missing user ID."""
        jwt_payload = {"role": "user", "email": "user@example.com"}

        with pytest.raises(HTTPException) as exc_info:
            await requires_user(jwt_payload)
        assert exc_info.value.status_code == 401
        assert "User ID not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_requires_user_empty_sub(self):
        """Test requires_user with empty user ID."""
        jwt_payload = {"sub": "", "role": "user"}

        with pytest.raises(HTTPException) as exc_info:
            await requires_user(jwt_payload)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_requires_admin_user_with_regular_user(self):
        """Test requires_admin_user rejects regular users."""
        jwt_payload = {"sub": "user-123", "role": "user", "email": "user@example.com"}

        with pytest.raises(HTTPException) as exc_info:
            await requires_admin_user(jwt_payload)
        assert exc_info.value.status_code == 403
        assert "Admin access required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_requires_admin_user_missing_role(self):
        """Test requires_admin_user with missing role."""
        jwt_payload = {"sub": "user-123", "email": "user@example.com"}

        with pytest.raises(KeyError):
            await requires_admin_user(jwt_payload)

    @pytest.mark.asyncio
    async def test_get_user_id_with_valid_payload(self, mocker: MockerFixture):
        """Test get_user_id extracts user ID correctly."""
        request = Mock(spec=Request)
        request.headers = {}
        jwt_payload = {"sub": "user-id-xyz", "role": "user"}

        mocker.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        )
        user_id = await get_user_id(request, jwt_payload)
        assert user_id == "user-id-xyz"

    @pytest.mark.asyncio
    async def test_get_user_id_missing_sub(self):
        """Test get_user_id with missing user ID."""
        request = Mock(spec=Request)
        request.headers = {}
        jwt_payload = {"role": "user"}

        with pytest.raises(HTTPException) as exc_info:
            await get_user_id(request, jwt_payload)
        assert exc_info.value.status_code == 401
        assert "User ID not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_user_id_none_sub(self):
        """Test get_user_id with None user ID."""
        request = Mock(spec=Request)
        request.headers = {}
        jwt_payload = {"sub": None, "role": "user"}

        with pytest.raises(HTTPException) as exc_info:
            await get_user_id(request, jwt_payload)
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
    async def test_dependency_error_cases(
        self, payload, expected_error: str, admin_only: bool
    ):
        """Test that errors propagate correctly through dependencies."""
        # Import verify_user to test it directly since dependencies use FastAPI Security
        from autogpt_libs.auth.jwt_utils import verify_user

        with pytest.raises(HTTPException) as exc_info:
            verify_user(payload, admin_only=admin_only)
        assert expected_error in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_dependency_valid_user(self):
        """Test valid user case for dependency."""
        # Import verify_user to test it directly since dependencies use FastAPI Security
        from autogpt_libs.auth.jwt_utils import verify_user

        # Valid case
        user = verify_user({"sub": "user", "role": "user"}, admin_only=False)
        assert user.user_id == "user"


class TestAdminImpersonation:
    """Test suite for admin user impersonation functionality."""

    @pytest.mark.asyncio
    async def test_admin_impersonation_success(self, mocker: MockerFixture):
        """Test admin successfully impersonating another user."""
        request = Mock(spec=Request)
        request.headers = {"X-Act-As-User-Id": "target-user-123"}
        jwt_payload = {
            "sub": "admin-456",
            "role": "admin",
            "email": "admin@example.com",
        }

        # Mock verify_user to return admin user data
        mock_verify_user = mocker.patch("autogpt_libs.auth.dependencies.verify_user")
        mock_verify_user.return_value = Mock(
            user_id="admin-456", email="admin@example.com", role="admin"
        )

        # Mock logger to verify audit logging
        mock_logger = mocker.patch("autogpt_libs.auth.dependencies.logger")

        mocker.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        )

        user_id = await get_user_id(request, jwt_payload)

        # Should return the impersonated user ID
        assert user_id == "target-user-123"

        # Should log the impersonation attempt
        mock_logger.info.assert_called_once()
        log_call = mock_logger.info.call_args[0][0]
        assert "Admin impersonation:" in log_call
        assert "admin@example.com" in log_call
        assert "target-user-123" in log_call

    @pytest.mark.asyncio
    async def test_non_admin_impersonation_attempt(self, mocker: MockerFixture):
        """Test non-admin user attempting impersonation returns 403."""
        request = Mock(spec=Request)
        request.headers = {"X-Act-As-User-Id": "target-user-123"}
        jwt_payload = {
            "sub": "regular-user",
            "role": "user",
            "email": "user@example.com",
        }

        # Mock verify_user to return regular user data
        mock_verify_user = mocker.patch("autogpt_libs.auth.dependencies.verify_user")
        mock_verify_user.return_value = Mock(
            user_id="regular-user", email="user@example.com", role="user"
        )

        mocker.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_user_id(request, jwt_payload)

        assert exc_info.value.status_code == 403
        assert "Only admin users can impersonate other users" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_impersonation_empty_header(self, mocker: MockerFixture):
        """Test impersonation with empty header falls back to regular user ID."""
        request = Mock(spec=Request)
        request.headers = {"X-Act-As-User-Id": ""}
        jwt_payload = {
            "sub": "admin-456",
            "role": "admin",
            "email": "admin@example.com",
        }

        mocker.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        )

        user_id = await get_user_id(request, jwt_payload)

        # Should fall back to the admin's own user ID
        assert user_id == "admin-456"

    @pytest.mark.asyncio
    async def test_impersonation_missing_header(self, mocker: MockerFixture):
        """Test normal behavior when impersonation header is missing."""
        request = Mock(spec=Request)
        request.headers = {}  # No impersonation header
        jwt_payload = {
            "sub": "admin-456",
            "role": "admin",
            "email": "admin@example.com",
        }

        mocker.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        )

        user_id = await get_user_id(request, jwt_payload)

        # Should return the admin's own user ID
        assert user_id == "admin-456"

    @pytest.mark.asyncio
    async def test_impersonation_audit_logging_details(self, mocker: MockerFixture):
        """Test that impersonation audit logging includes all required details."""
        request = Mock(spec=Request)
        request.headers = {"X-Act-As-User-Id": "victim-user-789"}
        jwt_payload = {
            "sub": "admin-999",
            "role": "admin",
            "email": "superadmin@company.com",
        }

        # Mock verify_user to return admin user data
        mock_verify_user = mocker.patch("autogpt_libs.auth.dependencies.verify_user")
        mock_verify_user.return_value = Mock(
            user_id="admin-999", email="superadmin@company.com", role="admin"
        )

        # Mock logger to capture audit trail
        mock_logger = mocker.patch("autogpt_libs.auth.dependencies.logger")

        mocker.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        )

        user_id = await get_user_id(request, jwt_payload)

        # Verify all audit details are logged
        assert user_id == "victim-user-789"
        mock_logger.info.assert_called_once()

        log_message = mock_logger.info.call_args[0][0]
        assert "Admin impersonation:" in log_message
        assert "superadmin@company.com" in log_message
        assert "victim-user-789" in log_message

    @pytest.mark.asyncio
    async def test_impersonation_header_case_sensitivity(self, mocker: MockerFixture):
        """Test that impersonation header is case-sensitive."""
        request = Mock(spec=Request)
        # Use wrong case - should not trigger impersonation
        request.headers = {"x-act-as-user-id": "target-user-123"}
        jwt_payload = {
            "sub": "admin-456",
            "role": "admin",
            "email": "admin@example.com",
        }

        mocker.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        )

        user_id = await get_user_id(request, jwt_payload)

        # Should fall back to admin's own ID (header case mismatch)
        assert user_id == "admin-456"

    @pytest.mark.asyncio
    async def test_impersonation_with_whitespace_header(self, mocker: MockerFixture):
        """Test impersonation with whitespace in header value."""
        request = Mock(spec=Request)
        request.headers = {"X-Act-As-User-Id": "  target-user-123  "}
        jwt_payload = {
            "sub": "admin-456",
            "role": "admin",
            "email": "admin@example.com",
        }

        # Mock verify_user to return admin user data
        mock_verify_user = mocker.patch("autogpt_libs.auth.dependencies.verify_user")
        mock_verify_user.return_value = Mock(
            user_id="admin-456", email="admin@example.com", role="admin"
        )

        # Mock logger
        mock_logger = mocker.patch("autogpt_libs.auth.dependencies.logger")

        mocker.patch(
            "autogpt_libs.auth.dependencies.get_jwt_payload", return_value=jwt_payload
        )

        user_id = await get_user_id(request, jwt_payload)

        # Should strip whitespace and impersonate successfully
        assert user_id == "target-user-123"
        mock_logger.info.assert_called_once()
