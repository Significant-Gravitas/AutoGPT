"""Fixtures shared by every copilot test."""

from typing import Any

import pytest

from backend.copilot.gate.reads import located_parts
from backend.copilot.tools.models import ToolResponseBase


@pytest.fixture(autouse=True)
def _declared_outside_parts_are_carried(request, monkeypatch):
    """A response that declares an outside part it does not carry is judged
    whole, the platform's own words included. Tests of that fallback opt out
    with ``@pytest.mark.absent_outside_part``."""
    if request.node.get_closest_marker("absent_outside_part"):
        return
    declare = ToolResponseBase.from_outside

    def checked(self: ToolResponseBase, *parts: Any) -> ToolResponseBase:
        full = self.model_dump_json(exclude_none=True)
        assert located_parts(parts, full, full) is not None, (
            f"{type(self).__name__} declares outside parts it does not carry: "
            f"{parts!r:.300}"
        )
        return declare(self, *parts)

    monkeypatch.setattr(ToolResponseBase, "from_outside", checked)


@pytest.fixture(autouse=True)
def _no_bans(mocker):
    """No owner is banned or closed unless a test says so. Without this the
    lookup goes to a DatabaseManager these tests don't run, and waits out its
    retries."""
    for module in (
        "backend.copilot.turn_queue",
        "backend.copilot.briefing.generate",
    ):
        mocker.patch(
            f"{module}.is_user_refused", new=mocker.AsyncMock(return_value=False)
        )
    mocker.patch(
        "backend.copilot.executor.processor.account_refusal",
        new=mocker.AsyncMock(return_value=None),
    )


@pytest.fixture(autouse=True)
def _no_cli_prelaunch(mocker):
    """No turn spawns a real CLI ahead of its query unless a test says so: a
    mocked ``ClaudeSDKClient`` would take the process and never stop it.
    Tests of the prelaunch patch ``service.CLIPrelaunch`` back themselves."""
    mocker.patch("backend.copilot.sdk.service.CLIPrelaunch", _NoCLIPrelaunch)


class _NoCLIPrelaunch:
    @staticmethod
    async def start(options, log_prefix):
        return None
