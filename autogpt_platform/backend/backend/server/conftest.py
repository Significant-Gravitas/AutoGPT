"""Common test fixtures for server tests."""

import pytest
from pytest_snapshot.plugin import Snapshot


@pytest.fixture
def configured_snapshot(snapshot: Snapshot) -> Snapshot:
    """Pre-configured snapshot fixture with standard settings."""
    snapshot.snapshot_dir = "snapshots"
    return snapshot


# Test ID constants
TEST_USER_ID = "test-user-id"
ADMIN_USER_ID = "admin-user-id"
TARGET_USER_ID = "target-user-id"


@pytest.fixture
def mock_jwt_user():
    """Provide mock JWT payload for regular user testing."""
    import fastapi

    def override_get_jwt_payload(request: fastapi.Request) -> dict[str, str]:
        return {"sub": TEST_USER_ID, "role": "user", "email": "test@example.com"}

    return {"get_jwt_payload": override_get_jwt_payload, "user_id": TEST_USER_ID}


@pytest.fixture
def mock_jwt_admin():
    """Provide mock JWT payload for admin user testing."""
    import fastapi

    def override_get_jwt_payload(request: fastapi.Request) -> dict[str, str]:
        return {
            "sub": ADMIN_USER_ID,
            "role": "admin",
            "email": "test-admin@example.com",
        }

    return {"get_jwt_payload": override_get_jwt_payload, "user_id": ADMIN_USER_ID}
