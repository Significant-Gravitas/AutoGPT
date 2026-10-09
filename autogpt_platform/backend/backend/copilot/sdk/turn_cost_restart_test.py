"""Cost accounting across the building-mode CLI interruption and relaunch."""

import contextlib
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamError
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.transcript import cli_session_path

from .conftest import build_cli_cost_row, build_test_transcript
from .retry_scenarios_test import _make_sdk_patches
from .turn_cost_test import _CALL_COST, _SDK_CWD, _SESSION_ID, _SVC, _result


@pytest.mark.asyncio
@pytest.mark.parametrize("result_before_restart", [False, True])
async def test_building_restart_counts_both_processes_once(
    result_before_restart, tmp_path, monkeypatch
):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    session_file = Path(cli_session_path(_SDK_CWD, _SESSION_ID))
    session_file.parent.mkdir(parents=True)
    transcript = build_test_transcript(
        [("user", "prior question"), ("assistant", "prior answer")]
    ) + build_cli_cost_row(_SESSION_ID, _CALL_COST)
    session = ChatSession(
        session_id=_SESSION_ID,
        user_id="test-user",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[
            ChatMessage(role="user", content="prior question"),
            ChatMessage(role="assistant", content="prior answer"),
            ChatMessage(role="user", content="build an agent"),
        ],
    )
    launched = []

    def client_factory(*args, **kwargs):
        launched.append(kwargs["options"])
        first_process = len(launched) == 1

        async def receive():
            if first_process:
                with session_file.open("a") as handle:
                    handle.write(build_cli_cost_row(_SESSION_ID, 2 * _CALL_COST))
                session.building_mode_requested = not result_before_restart
            yield AssistantMessage(
                content=[TextBlock(text="ok")], model="claude-sonnet-4-6"
            )
            if first_process:
                session.building_mode_requested = True
            yield _result((2 if first_process else 3) * _CALL_COST)

        client = MagicMock()
        client.query = AsyncMock()
        client.interrupt = AsyncMock()
        client.receive_response = receive
        context = AsyncMock()
        context.__aenter__.return_value = client
        context.__aexit__.return_value = None
        return context

    patches = _make_sdk_patches(
        session,
        original_transcript=transcript,
        compacted_transcript=None,
        client_side_effect=client_factory,
    )
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        stack.enter_context(
            patch(
                f"{_SVC}.build_skills_update_notice",
                new_callable=AsyncMock,
                return_value=None,
            )
        )
        stack.enter_context(
            patch(
                f"{_SVC}.build_builder_system_prompt_suffix",
                new_callable=AsyncMock,
                side_effect=["", "<building_guide>guide</building_guide>"],
            )
        )
        persist = stack.enter_context(
            patch(f"{_SVC}.persist_and_record_usage", new_callable=AsyncMock)
        )
        events = [
            event
            async for event in stream_chat_completion_sdk(
                session_id=_SESSION_ID,
                message="build an agent",
                is_user_message=True,
                user_id="test-user",
                session=session,
            )
        ]

    assert [options.resume for options in launched] == [_SESSION_ID, _SESSION_ID]
    assert session.guide_in_system_prompt
    assert not any(isinstance(event, StreamError) for event in events)
    persist.assert_awaited_once()
    assert persist.await_args.kwargs["cost_usd"] == pytest.approx(2 * _CALL_COST)
