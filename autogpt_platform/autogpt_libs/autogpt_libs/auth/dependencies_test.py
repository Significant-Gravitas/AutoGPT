"""
Comprehensive integration tests for authentication dependencies.
Tests the full authentication flow from HTTP requests to user validation.
"""

import os
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from autogpt_libs.auth import config
from autogpt_libs.auth.config import Settings
from autogpt_libs.auth.dependencies import (
    _ensure_platform_user,
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
        """A missing 'role' claim must fail closed (403), not raise KeyError."""
        jwt_payload = {"sub": "user-123", "email": "user@example.com"}

        with pytest.raises(HTTPException) as exc_info:
            await requires_admin_user(jwt_payload)
        assert exc_info.value.status_code == 403
        assert "Admin access required" in exc_info.value.detail

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

        # JWT_JWKS_URL is required by Settings.validate(); HS256 tokens verify
        # against JWT_VERIFY_KEY and never touch the JWKS client, so a
        # present-but-unused URL is enough. Reset the cached settings so
        # get_settings() rebuilds under this patched environment.
        mocker.patch.dict(
            os.environ,
            {
                "JWT_VERIFY_KEY": self.acceptable_jwt_secret,
                "JWT_JWKS_URL": "http://localhost:3000/api/auth/jwks",
            },
            clear=True,
        )
        mocker.patch.object(config, "_settings", Settings())

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


class TestEnsurePlatformUser:
    """A valid token whose platform User row is missing must self-heal.

    Better Auth issues a session as soon as the auth identity exists, so a
    request can arrive before (or without) the client-driven provisioning
    call. Before this, the org bootstrap could not create an org for a user
    it could not find and the account 400'd on every org-scoped endpoint
    forever — see BUILDER-50B.
    """

    @staticmethod
    def _stub_backend(mocker: MockerFixture, *, existing_user, provisioner):
        """Point the deferred `backend.*` imports at test doubles."""
        import sys
        import types

        db_mod = types.ModuleType("backend.data.db")
        db_mod.prisma = Mock()
        # A list means "successive calls" -- used to model the row appearing
        # between the initial probe and the post-failure re-check.
        db_mod.prisma.user.find_unique = (
            AsyncMock(side_effect=list(existing_user))
            if isinstance(existing_user, list)
            else AsyncMock(return_value=existing_user)
        )

        user_mod = types.ModuleType("backend.data.user")
        user_mod.get_or_create_user_with_status = provisioner

        mocker.patch.dict(
            sys.modules,
            {
                "backend": types.ModuleType("backend"),
                "backend.data": types.ModuleType("backend.data"),
                "backend.data.db": db_mod,
                "backend.data.user": user_mod,
            },
        )
        return db_mod

    @pytest.mark.asyncio
    async def test_provisions_when_user_row_missing(self, mocker: MockerFixture):
        provision = AsyncMock()
        self._stub_backend(mocker, existing_user=None, provisioner=provision)
        payload = {"sub": "user-1", "email": "new@example.com"}

        await _ensure_platform_user("user-1", payload)

        provision.assert_awaited_once_with(payload)

    @pytest.mark.asyncio
    async def test_noop_when_user_row_exists(self, mocker: MockerFixture):
        provision = AsyncMock()
        self._stub_backend(mocker, existing_user=Mock(), provisioner=provision)

        await _ensure_platform_user("user-1", {"sub": "user-1", "email": "a@b.c"})

        provision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_never_provisions_from_an_impersonators_claims(
        self, mocker: MockerFixture
    ):
        """Under impersonation the JWT describes the admin, not the target.

        Provisioning from it would create the target's account under the
        admin's email, so the self-heal must decline entirely.
        """
        provision = AsyncMock()
        self._stub_backend(mocker, existing_user=None, provisioner=provision)

        await _ensure_platform_user(
            "target-user", {"sub": "admin-456", "email": "admin@example.com"}
        )

        provision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_token_carries_no_email(self, mocker: MockerFixture):
        logger = mocker.patch("autogpt_libs.auth.dependencies.logger")
        provision = AsyncMock()
        self._stub_backend(mocker, existing_user=None, provisioner=provision)

        await _ensure_platform_user("user-1", {"sub": "user-1"})

        provision.assert_not_awaited()
        # This account stays broken, so the refusal must not be silent —
        # bricked *and* invisible is the failure mode this function ends.
        logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_provisioning_failure_is_swallowed(self, mocker: MockerFixture):
        """Best-effort: the caller still runs the org bootstrap and surfaces
        the same 400 as before, so a failure here can only be neutral."""
        provision = AsyncMock(side_effect=RuntimeError("db down"))
        self._stub_backend(mocker, existing_user=None, provisioner=provision)

        await _ensure_platform_user("user-1", {"sub": "user-1", "email": "a@b.c"})

        provision.assert_awaited_once()


class TestRequestContextProvisioning:
    """`get_request_context` must actually invoke the self-heal.

    The whole effect of the fix lives on one call site; without this, deleting
    that line leaves the unit tests above green.
    """

    @staticmethod
    def _request(headers: dict | None = None):
        request = Mock(spec=Request)
        request.headers = headers or {}
        request.method = "GET"
        request.url = "http://test/api/library/agents"
        return request

    @pytest.fixture
    def wiring(self, mocker: MockerFixture):
        """Stub the deferred backend imports and record call order."""
        import sys
        import types

        calls: list[str] = []

        org_member = Mock(
            status="ACTIVE", isOwner=True, isAdmin=False, isBillingManager=False
        )
        org_member.Org = Mock(deletedAt=None)

        db_mod = types.ModuleType("backend.data.db")
        db_mod.prisma = Mock()
        # No personal org -> the self-heal branch.
        db_mod.prisma.orgmember.find_first = AsyncMock(return_value=None)
        db_mod.prisma.orgmember.find_unique = AsyncMock(return_value=org_member)

        async def _default_team(_user_id):
            calls.append("get_user_default_team")
            return "org-1", "team-1"

        orgs_mod = types.ModuleType("backend.api.features.orgs.db")
        orgs_mod.get_user_default_team = _default_team

        mocker.patch.dict(
            sys.modules,
            {
                "backend": types.ModuleType("backend"),
                "backend.data": types.ModuleType("backend.data"),
                "backend.data.db": db_mod,
                "backend.api": types.ModuleType("backend.api"),
                "backend.api.features": types.ModuleType("backend.api.features"),
                "backend.api.features.orgs": types.ModuleType(
                    "backend.api.features.orgs"
                ),
                "backend.api.features.orgs.db": orgs_mod,
            },
        )

        async def _ensure(user_id, payload):
            calls.append("ensure_platform_user")

        ensure = mocker.patch(
            "autogpt_libs.auth.dependencies._ensure_platform_user",
            side_effect=_ensure,
        )
        return ensure, calls

    @pytest.mark.asyncio
    async def test_self_heal_runs_before_the_org_bootstrap(self, wiring):
        """Order matters: provisioning after the bootstrap would be useless,
        since the bootstrap is what needs the User row to exist."""
        from autogpt_libs.auth.dependencies import get_request_context

        ensure, calls = wiring
        payload = {"sub": "user-1", "email": "new@example.com"}

        ctx = await get_request_context(self._request(), payload)

        ensure.assert_awaited_once_with("user-1", payload)
        assert calls == ["ensure_platform_user", "get_user_default_team"]
        assert ctx.org_id == "org-1"

    @pytest.mark.asyncio
    async def test_skipped_when_the_user_already_has_a_personal_org(
        self, wiring, mocker: MockerFixture
    ):
        """The common path must not pay for the self-heal."""
        import sys

        from autogpt_libs.auth.dependencies import get_request_context

        ensure, _ = wiring
        member = Mock(orgId="org-existing")
        sys.modules["backend.data.db"].prisma.orgmember.find_first = AsyncMock(
            return_value=member
        )

        await get_request_context(self._request(), {"sub": "user-1", "email": "a@b.c"})

        ensure.assert_not_awaited()


class TestEnsurePlatformUserRaceLogging:
    """A losing create race must not be reported as a failure.

    One first page load fans out ~20 requests that all miss the probe, so all
    but one lose the race inside `get_or_create_user_with_status` (Prisma's
    UniqueViolationError, rewrapped as DatabaseError). Logging those at ERROR
    put ~19 tracebacks claiming failure behind every successful heal, which
    would make Sentry actively misleading about whether the fix works.
    """

    @pytest.mark.asyncio
    async def test_lost_race_is_not_reported_as_a_failure(self, mocker: MockerFixture):
        logger = mocker.patch("autogpt_libs.auth.dependencies.logger")
        # Missing on the probe, present on the post-failure re-check.
        TestEnsurePlatformUser._stub_backend(
            mocker,
            existing_user=[None, Mock()],
            provisioner=AsyncMock(side_effect=RuntimeError("unique violation")),
        )

        await _ensure_platform_user("user-1", {"sub": "user-1", "email": "a@b.c"})

        logger.error.assert_not_called()
        # The traceback still has to survive somewhere: WARNING is a Sentry
        # breadcrumb, not an event, so the detail is kept without the noise.
        logger.warning.assert_called_once()
        assert logger.warning.call_args.kwargs.get("exc_info") is True

    @pytest.mark.asyncio
    async def test_genuine_failure_is_still_reported(self, mocker: MockerFixture):
        logger = mocker.patch("autogpt_libs.auth.dependencies.logger")
        # Still missing on the re-check -- the user really is unprovisioned.
        TestEnsurePlatformUser._stub_backend(
            mocker,
            existing_user=[None, None],
            provisioner=AsyncMock(side_effect=RuntimeError("db down")),
        )

        await _ensure_platform_user("user-1", {"sub": "user-1", "email": "a@b.c"})

        logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_heal_reports_once(self, mocker: MockerFixture):
        """The invariant breach is worth one Sentry event per account."""
        logger = mocker.patch("autogpt_libs.auth.dependencies.logger")
        TestEnsurePlatformUser._stub_backend(
            mocker,
            existing_user=None,
            provisioner=AsyncMock(return_value=Mock(was_created=True)),
        )

        await _ensure_platform_user("user-1", {"sub": "user-1", "email": "a@b.c"})

        logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_row_found_rather_than_created_is_not_reported(
        self, mocker: MockerFixture
    ):
        """Returning without raising is not proof we created anything.

        A concurrent request can land its row between our probe and the
        get-or-create's own lookup, which then reads it back with
        was_created=False. Reporting that too would put several ERRORs behind
        one heal, each claiming to have provisioned a row it only found.
        """
        logger = mocker.patch("autogpt_libs.auth.dependencies.logger")
        TestEnsurePlatformUser._stub_backend(
            mocker,
            existing_user=None,
            provisioner=AsyncMock(return_value=Mock(was_created=False)),
        )

        await _ensure_platform_user("user-1", {"sub": "user-1", "email": "a@b.c"})

        logger.error.assert_not_called()
