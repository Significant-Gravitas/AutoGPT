"""Common test fixtures for server tests.

Note: Common fixtures like test_user_id, admin_user_id, target_user_id,
setup_test_user, and setup_admin_user are defined in the parent conftest.py
(backend/conftest.py) and are available here automatically.
"""

import pytest
from pytest_snapshot.plugin import Snapshot


@pytest.fixture
def test_user_id() -> str:
    return "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


@pytest.fixture
def admin_user_id() -> str:
    return "4e53486c-cf57-477e-ba2a-cb02dc828e1b"


@pytest.fixture
def setup_test_user(test_user_id: str) -> str:
    return test_user_id


@pytest.fixture
def configured_snapshot(snapshot: Snapshot) -> Snapshot:
    """Pre-configured snapshot fixture with standard settings."""
    snapshot.snapshot_dir = "snapshots"
    return snapshot


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
