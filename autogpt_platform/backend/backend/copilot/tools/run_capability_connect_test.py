"""``connect: true`` surfaces or confirms credentials without running anything."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.copilot.capabilities.models import (
    CapabilityEntry,
    Connection,
    Implementation,
)
from backend.copilot.gate.subject import NO_OP
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import (
    CapabilityDetailsResponse,
    SetupInfo,
    SetupRequirementsResponse,
    UserReadiness,
)
from backend.copilot.tools.run_capability import RunCapabilityTool, _run_block

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


def _card(
    session_id: str, has_all_credentials: bool = False
) -> SetupRequirementsResponse:
    return SetupRequirementsResponse(
        message="needs linear",
        session_id=session_id,
        setup_info=SetupInfo(
            agent_id=BLOCK_ID,
            agent_name="LinearCreateIssueBlock",
            user_readiness=UserReadiness(
                has_all_credentials=has_all_credentials,
                missing_credentials={},
                ready_to_run=False,
            ),
            requirements={
                "credentials": [],
                "inputs": [],
                "execution_modes": ["immediate"],
            },
        ),
    )


async def _connect(prep_result, validate_only: bool = False):
    session = make_session(USER)
    prepare = AsyncMock(return_value=prep_result(session.session_id))
    with (
        patch("backend.copilot.tools.run_capability.gate_denied", return_value=False),
        patch(
            "backend.copilot.tools.run_capability.prepare_block_for_execution",
            prepare,
        ),
        patch("backend.copilot.tools.run_capability.RunBlockTool") as run_block,
    ):
        result = await _run_block(
            _entry(), USER, session, {"connect": True}, validate_only, False
        )
    prepare.assert_awaited_once()
    assert "connect" not in prepare.await_args.kwargs["input_data"]
    run_block.assert_not_called()
    return result, prepare


async def test_connect_returns_the_card_when_credentials_are_missing():
    result, _ = await _connect(_card)
    assert isinstance(result, SetupRequirementsResponse)


async def test_connect_confirms_and_does_not_run_when_credentials_exist():
    # any non-ToolResponseBase stands for a ready preparation
    result, _ = await _connect(lambda _session_id: object())
    assert isinstance(result, CapabilityDetailsResponse)
    assert "connected" in result.message.lower()
    assert "nothing was run" in result.message.lower()


async def test_connect_treats_a_picker_only_card_as_connected():
    result, _ = await _connect(lambda sid: _card(sid, has_all_credentials=True))
    assert isinstance(result, CapabilityDetailsResponse)
    assert "connected" in result.message.lower()


async def test_connect_passes_validate_only_through():
    _, prepare = await _connect(lambda _session_id: object(), validate_only=True)
    assert prepare.await_args.kwargs["validate_only"] is True


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


async def test_a_connect_call_on_a_block_has_no_gate_subject():
    session = make_session(USER)
    entry = _entry()
    with (
        patch(
            "backend.copilot.tools.run_capability.resolve_session_entry",
            AsyncMock(return_value=entry),
        ),
        patch(
            "backend.copilot.tools.run_capability.get_block", return_value=MagicMock()
        ),
        patch(
            "backend.copilot.tools.run_capability.required_input_keys",
            return_value=set(),
        ),
        patch(
            "backend.copilot.tools.run_capability.block_subject",
            return_value=SimpleNamespace(effect=None),
        ),
    ):
        subject = await RunCapabilityTool().gate_subject(
            USER, session, {"id": entry.id, "input": {"connect": True}}
        )
    assert subject is NO_OP
