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
    return "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


@pytest.fixture
def admin_user_id() -> str:
    """Admin user ID fixture."""
    return "4e53486c-cf57-477e-ba2a-cb02dc828e1b"


@pytest.fixture
def target_user_id() -> str:
    """Target user ID fixture."""
    return "5e53486c-cf57-477e-ba2a-cb02dc828e1c"


@pytest.fixture
async def setup_test_user(test_user_id):
    """Create test user in database before tests."""
    from backend.data.user import get_or_create_user

    # Create the test user in the database using JWT token format
    user_data = {
        "sub": test_user_id,
        "email": "test@example.com",
        "user_metadata": {"name": "Test User"},
    }
    await get_or_create_user(user_data)
    return test_user_id


@pytest.fixture
async def setup_admin_user(admin_user_id):
    """Create admin user in database before tests."""
    from backend.data.user import get_or_create_user

    # Create the admin user in the database using JWT token format
    user_data = {
        "sub": admin_user_id,
        "email": "test-admin@example.com",
        "user_metadata": {"name": "Test Admin"},
    }
    await get_or_create_user(user_data)
    return admin_user_id


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
