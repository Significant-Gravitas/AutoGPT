"""Override session-scoped fixtures so unit tests run without the server."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="session")
def server():
    yield None


@pytest.fixture(autouse=True)
def resume_after_reconnect(mocker) -> MagicMock:
    """A connect resumes the owner's paused scheduled chats in the background.
    Without the server there are none, so the call is only recorded."""
    return mocker.patch(
        "backend.api.features.integrations.router.resume_after_reconnect_in_background"
    )
