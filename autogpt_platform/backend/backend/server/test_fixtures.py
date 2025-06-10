"""Common test fixtures with proper setup and teardown."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from unittest.mock import Mock, patch

import pytest
from prisma import Prisma


@pytest.fixture
async def test_db_connection() -> AsyncGenerator[Prisma, None]:
    """Provide a test database connection with proper cleanup.

    This fixture ensures the database connection is properly
    closed after the test, even if the test fails.
    """
    db = Prisma()
    try:
        await db.connect()
        yield db
    finally:
        await db.disconnect()


@pytest.fixture
def mock_transaction():
    """Mock database transaction with proper async context manager."""

    @asynccontextmanager
    async def mock_context(*args, **kwargs):
        yield None

    with patch("backend.data.db.locked_transaction", side_effect=mock_context) as mock:
        yield mock


@pytest.fixture
def isolated_app_state():
    """Fixture that ensures app state is isolated between tests."""
    # Example: Save original state
    # from backend.server.app import app
    # original_overrides = app.dependency_overrides.copy()

    # try:
    #     yield app
    # finally:
    #     # Restore original state
    #     app.dependency_overrides = original_overrides

    # For now, just yield None as this is an example
    yield None


@pytest.fixture
def cleanup_files():
    """Fixture to track and cleanup files created during tests."""
    created_files = []

    def track_file(filepath: str):
        created_files.append(filepath)

    yield track_file

    # Cleanup
    import os

    for filepath in created_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Warning: Failed to cleanup {filepath}: {e}")


@pytest.fixture
async def async_mock_with_cleanup():
    """Create async mocks that are properly cleaned up."""
    mocks = []

    def create_mock(**kwargs):
        mock = Mock(**kwargs)
        mocks.append(mock)
        return mock

    yield create_mock

    # Reset all mocks
    for mock in mocks:
        mock.reset_mock()


class TestDatabaseIsolation:
    """Example of proper test isolation with database operations."""

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self, test_db_connection):
        """Setup and teardown for each test method."""
        # Setup: Clear test data
        await test_db_connection.user.delete_many(
            where={"email": {"contains": "@test.example"}}
        )

        yield

        # Teardown: Clear test data again
        await test_db_connection.user.delete_many(
            where={"email": {"contains": "@test.example"}}
        )

    async def test_create_user(self, test_db_connection):
        """Test that demonstrates proper isolation."""
        # This test has access to a clean database
        user = await test_db_connection.user.create(
            data={"email": "test@test.example", "name": "Test User"}
        )
        assert user.email == "test@test.example"
        # User will be cleaned up automatically


@pytest.fixture(scope="function")  # Explicitly use function scope
def reset_singleton_state():
    """Reset singleton state between tests."""
    # Example: Reset a singleton instance
    # from backend.data.some_singleton import SingletonClass

    # # Save original state
    # original_instance = getattr(SingletonClass, "_instance", None)

    # try:
    #     # Clear singleton
    #     SingletonClass._instance = None
    #     yield
    # finally:
    #     # Restore original state
    #     SingletonClass._instance = original_instance

    # For now, just yield None as this is an example
    yield None
