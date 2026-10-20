"""Fixtures shared by every copilot test."""

import pytest


@pytest.fixture(autouse=True)
def _no_bans(mocker):
    """No owner is banned unless a test says so. Without this the lookup goes
    to a DatabaseManager these tests don't run, and waits out its retries."""
    for module in (
        "backend.copilot.turn_queue",
        "backend.copilot.executor.processor",
        "backend.copilot.briefing.generate",
    ):
        mocker.patch(
            f"{module}.is_user_banned", new=mocker.AsyncMock(return_value=False)
        )
