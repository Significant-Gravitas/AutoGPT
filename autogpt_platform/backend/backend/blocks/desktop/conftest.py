"""Shared fixtures for the desktop tests."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def login_chain_unchanged(request):
    """Sandbox doubles hold no login files; tests of that check opt out."""
    if request.node.get_closest_marker("real_login_chain"):
        yield
        return
    unchanged = AsyncMock(return_value={})
    with (
        patch("backend.util.sandbox_login.changed_login_files", unchanged),
        patch("backend.blocks.desktop._api.take_baseline", AsyncMock()),
    ):
        yield
