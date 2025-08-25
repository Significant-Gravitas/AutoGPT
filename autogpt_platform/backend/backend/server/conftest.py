"""Common test fixtures for server tests."""

import pytest
from pytest_snapshot.plugin import Snapshot


@pytest.fixture
def configured_snapshot(snapshot: Snapshot) -> Snapshot:
    """Pre-configured snapshot fixture with standard settings."""
    snapshot.snapshot_dir = "snapshots"
    return snapshot


@pytest.fixture
def test_user_id() -> str:
    """Test user ID fixture."""
    return "test-user-id"


@pytest.fixture
def admin_user_id() -> str:
    """Admin user ID fixture."""
    return "admin-user-id"


@pytest.fixture
def target_user_id() -> str:
    """Target user ID fixture."""
    return "target-user-id"


@pytest.fixture
def mock_jwt_user(test_user_id):
    """Provide mock JWT payload for regular user testing."""
    import fastapi

    def override_get_jwt_payload(request: fastapi.Request) -> dict[str, str]:
        return {"sub": test_user_id, "role": "user", "email": "test@example.com"}

    return {"get_jwt_payload": override_get_jwt_payload, "user_id": test_user_id}


@pytest.fixture
def mock_jwt_admin(admin_user_id):
    """Provide mock JWT payload for admin user testing."""
    import fastapi

    def override_get_jwt_payload(request: fastapi.Request) -> dict[str, str]:
        return {
            "sub": admin_user_id,
            "role": "admin",
            "email": "test-admin@example.com",
        }

    return {"get_jwt_payload": override_get_jwt_payload, "user_id": admin_user_id}
