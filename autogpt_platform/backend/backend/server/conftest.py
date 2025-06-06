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
