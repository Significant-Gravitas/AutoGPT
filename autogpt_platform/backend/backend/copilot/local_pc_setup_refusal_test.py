"""A Local PC chat whose turn gets no computer says which of two things is
wrong: the chat's folder there can no longer be used, or the computer is not
connected. Only the second is fixed by starting the executor.

The input is what the computer answers on its control channel; the output is
what the turn tells the person.
"""

import contextlib
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.constants import COPILOT_ERROR_PREFIX
from backend.copilot.model import ChatSession, LocalExecutionTargetMetadata
from backend.copilot.response_model import StreamError, StreamStart
from backend.copilot.sdk import retry_scenarios_test as scenarios
from backend.copilot.sdk import service
from backend.copilot.sdk.cli_prelaunch_turn_test import _patched_turn
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.tools.local_pc_relay_protocol import (
    RelayPresence,
    machine_scope_id,
)

_FOLDER_CHANGED = StreamError(
    errorText=(
        "The folder this chat uses on your computer was moved or replaced, or "
        "can't be reached, so Otto can't work in it. Start a new chat and pick "
        "the folder again."
    ),
    code="local_pc_folder_changed",
)
_NOT_CONNECTED = StreamError(
    errorText=(
        "Your Local PC executor is not connected. Start it for this chat "
        "session and try again."
    ),
    code="local_pc_executor_unavailable",
)
# The first thing a turn does once its executor is set up: a turn that went on
# without one fails here at once instead of running on into the unmocked rest.
_WENT_ON = AssertionError("the turn went on without its computer")
_USER = "test-user"
_MACHINE = "machine-1"
_CLIENT = "autogpt-local-executor"


class _Computer:
    """One computer as the relay sees it: connected or not, and what it
    answers when asked to restore a chat's folder."""

    def __init__(self, restore_reply: dict[str, Any] | None, *, connected: bool = True):
        self.restore_reply = restore_reply
        self.connected = connected
        self.asked: list[str] = []

    async def get_presence(self, _scope: str) -> RelayPresence | None:
        if not self.connected:
            return None
        return RelayPresence(
            session_id=machine_scope_id(_USER, _MACHINE),
            connection_id="connection-1",
            user_id=_USER,
            client_id=_CLIENT,
            hello={"connection_kind": "machine", "machine_id": _MACHINE},
            expires_at=9_999_999_999,
        )

    async def open_transport(self, _presence: RelayPresence) -> "_Channel":
        return _Channel(self)


class _Channel:
    def __init__(self, computer: _Computer) -> None:
        self._computer = computer
        self._request: dict[str, Any] = {}

    async def send_text(self, envelope: str) -> None:
        self._request = json.loads(envelope)
        self._computer.asked.append(self._request["type"])

    async def iter_text(self):
        reply = (
            {"type": "ACK", "payload": {"ok": True}}
            if self._request["type"] == "DETACH_SESSION"
            else self._computer.restore_reply
        )
        if reply is None:
            raise ConnectionError("the computer went away")
        yield json.dumps({"id": self._request["id"], **reply})

    async def close(self) -> None:
        return None


def _refusal(code: str, message: str) -> dict[str, Any]:
    return {"type": "ERROR", "payload": {"code": code, "message": message}}


async def _turn_on(
    computer: _Computer, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[StreamError], list[dict], ChatSession]:
    """Run one turn of a chat bound to a folder on *computer*; return its
    errors, the CLI launches and the session."""
    monkeypatch.setenv("FORCE_FLAG_LOCAL_PC_EXECUTOR", "true")
    turns = scenarios.TestStreamChatCompletionRetryIntegration()
    launches: list[dict] = []

    def client(*_args, **kwargs):
        launches.append(kwargs)
        return turns._make_client_mock(result_message=turns._make_result_message())

    session = turns._make_session()
    session.metadata = session.metadata.model_copy(
        update={
            "execution_target": LocalExecutionTargetMetadata(
                machine_id=_MACHINE,
                directory_ref="directory-1",
                allowed_root="/Users/ada/Projects",
                root_fingerprint="a" * 64,
                root_grant="grant-1",
                revision=1,
            )
        }
    )
    no_data_channel = MagicMock()
    no_data_channel.get_hello_async = AsyncMock(return_value=None)
    events = []
    with contextlib.ExitStack() as stack:
        stack.enter_context(_patched_turn(client, session))
        stack.enter_context(
            patch.object(service.config, "local_pc_executor_oauth_client_id", _CLIENT)
        )
        stack.enter_context(
            patch.object(service, "get_shim_manager", return_value=no_data_channel)
        )
        stack.enter_context(
            patch(
                "backend.copilot.tools.local_pc_machine.get_local_pc_relay",
                return_value=computer,
            )
        )
        stack.enter_context(
            patch.object(
                service, "approval_mode_supplement", MagicMock(side_effect=_WENT_ON)
            )
        )
        async for event in stream_chat_completion_sdk(
            session_id=session.session_id,
            message="hello",
            is_user_message=True,
            user_id=_USER,
            session=session,
        ):
            events.append(event)

    assert not [e for e in events if isinstance(e, StreamStart)]
    return [e for e in events if isinstance(e, StreamError)], launches, session


async def test_a_refused_folder_grant_tells_the_person_to_start_a_new_chat(
    monkeypatch: pytest.MonkeyPatch,
):
    """The computer is connected and answers: it will not restore the chat's
    folder, because the folder was moved or replaced. The turn said "Your
    Local PC executor is not connected", and nothing in that chat fixes it,
    since a folder attaches only when a chat is created."""
    computer = _Computer(
        _refusal(
            "ROOT_GRANT_INVALID",
            "The selected directory changed after the grant was issued",
        )
    )

    errors, launches, session = await _turn_on(computer, monkeypatch)

    assert errors == [_FOLDER_CHANGED]
    assert launches == []
    # Saved with the chat, so it still says why after a reload.
    assert session.messages[-1].content == (
        f"{COPILOT_ERROR_PREFIX} {_FOLDER_CHANGED.errorText}"
    )
    assert computer.asked == ["RESTORE_SESSION", "DETACH_SESSION"]


@pytest.mark.parametrize(
    "computer",
    [
        _Computer(None, connected=False),
        _Computer(None),
        _Computer(
            _refusal(
                "SESSION_REVISION_MISMATCH",
                "The supplied root grant is older than the current binding.",
            )
        ),
    ],
    ids=["not-connected", "went-away-mid-request", "another-refusal"],
)
async def test_any_other_failure_still_says_the_computer_is_not_connected(
    computer: _Computer, monkeypatch: pytest.MonkeyPatch
):
    errors, launches, session = await _turn_on(computer, monkeypatch)

    assert errors == [_NOT_CONNECTED]
    assert launches == []
    assert COPILOT_ERROR_PREFIX not in (session.messages[-1].content or "")
