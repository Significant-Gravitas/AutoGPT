"""``connect: true`` surfaces or confirms credentials without running anything."""

from unittest.mock import AsyncMock, patch

from backend.copilot.capabilities.models import (
    CapabilityEntry,
    Connection,
    Implementation,
)
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import (
    CapabilityDetailsResponse,
    SetupInfo,
    SetupRequirementsResponse,
    UserReadiness,
)
from backend.copilot.tools.run_capability import _run_block

USER = "user-connect"
BLOCK_ID = "11111111-1111-1111-1111-111111111111"


def _entry() -> CapabilityEntry:
    return CapabilityEntry(
        id=f"block:{BLOCK_ID}",
        kind="block",
        name="LinearCreateIssueBlock",
        purpose="Create an issue.",
        implementations=[Implementation(kind="block", ref=BLOCK_ID)],
        connection=Connection(required=True, key_type="provider", key="linear"),
        service="linear",
    )


def _card(session_id: str) -> SetupRequirementsResponse:
    return SetupRequirementsResponse(
        message="needs linear",
        session_id=session_id,
        setup_info=SetupInfo(
            agent_id=BLOCK_ID,
            agent_name="LinearCreateIssueBlock",
            user_readiness=UserReadiness(
                has_all_credentials=False, missing_credentials={}, ready_to_run=False
            ),
            requirements={
                "credentials": [],
                "inputs": [],
                "execution_modes": ["immediate"],
            },
        ),
    )


async def test_connect_returns_the_card_when_credentials_are_missing():
    session = make_session(USER)
    with (
        patch("backend.copilot.tools.run_capability.gate_denied", return_value=False),
        patch(
            "backend.copilot.tools.run_capability.prepare_block_for_execution",
            AsyncMock(return_value=_card(session.session_id)),
        ),
        patch("backend.copilot.tools.run_capability.RunBlockTool") as run_block,
    ):
        result = await _run_block(
            _entry(), USER, session, {"connect": True}, False, False
        )
    assert isinstance(result, SetupRequirementsResponse)
    run_block.assert_not_called()


async def test_connect_confirms_and_does_not_run_when_credentials_exist():
    session = make_session(USER)
    prep = object()  # any non-ToolResponseBase stands for a ready preparation
    with (
        patch("backend.copilot.tools.run_capability.gate_denied", return_value=False),
        patch(
            "backend.copilot.tools.run_capability.prepare_block_for_execution",
            AsyncMock(return_value=prep),
        ),
        patch("backend.copilot.tools.run_capability.RunBlockTool") as run_block,
    ):
        result = await _run_block(
            _entry(), USER, session, {"connect": True}, False, False
        )
    assert isinstance(result, CapabilityDetailsResponse)
    assert "connected" in result.message.lower()
    assert "nothing was run" in result.message.lower()
    run_block.assert_not_called()


async def test_without_connect_the_block_runs_as_before():
    session = make_session(USER)
    with (
        patch("backend.copilot.tools.run_capability.gate_denied", return_value=False),
        patch("backend.copilot.tools.run_capability.RunBlockTool") as run_block,
    ):
        run_block.return_value._execute = AsyncMock(
            return_value=_card(session.session_id)
        )
        await _run_block(_entry(), USER, session, {"title": "x"}, False, False)
    run_block.return_value._execute.assert_awaited_once()
