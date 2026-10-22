"""Chat text reaches the model as the user wrote it, through whole copilot turns.

``stream_chat_completion_sdk`` drives the bundled Claude Code CLI against the
billing suite's fake provider, so the CLI options, the CLI version check, the
session upload and restore, and ``--resume`` are production's (see
``billing_e2e/sdk_turns.py`` for what is stubbed). Text that names files or
directories, or starts with a CLI command, must reach the provider as written,
and nothing those files hold may reach it, on that turn or any later one, or
be stored with the session.

The Linux SDK wheel bundles the CLI, so without one the tests fail there; on
other platforms, whose wheels ship without it, they skip. CI runs them on Linux.
A CLI release that writes what it attached into a kind of session line the
upload keeps fails them, so an SDK upgrade, which brings its own CLI, is
checked here.
"""

import json
import platform
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

from backend.copilot.billing_e2e.invoice import SONNET, Route
from backend.copilot.billing_e2e.ledger import PlatformLedger
from backend.copilot.billing_e2e.provider import FakeProvider
from backend.copilot.billing_e2e.sdk_turns import SdkTurns
from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.transcript import cli_session_path

from . import service as sdk_service
from .cli_openrouter_compat_test import _resolve_cli_path

_NOTE = "The note's own words."
# A line cut off mid-write, which the upload's strip cannot parse.
_CUT_OFF_LINE = b'{"type":"attachment","attachment":{"type":"file","con\n'
_READ_SESSION = sdk_service.read_cli_session_from_disk


@pytest.fixture
def ledger(monkeypatch: pytest.MonkeyPatch) -> PlatformLedger:
    """Takes the spend, cost-log and credit writes each turn makes."""
    ledger = PlatformLedger()
    ledger.install(monkeypatch)
    return ledger


@pytest.fixture(autouse=True)
def _memory_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory is not under test here, and its ingest needs a database."""
    monkeypatch.setattr(
        sdk_service, "_fetch_graphiti_context", AsyncMock(return_value=(False, ""))
    )


# Its own event loop, like the billing suite: a background task another test
# left on the shared session loop must not run into this one's turns.
@pytest.mark.asyncio(loop_scope="function")
async def test_chat_text_reaches_the_model_as_written(
    ledger: PlatformLedger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A fresh turn, a turn resumed on the same pod, and one resumed on a new
    pod from the stored session."""
    _require_cli()
    async with _RecordingProvider() as provider:
        turns = SdkTurns(tmp_path, monkeypatch, provider, _config())
        elsewhere = _write_files(tmp_path)
        prompts = [
            f"compare @note.txt with @{(elsewhere / 'note.txt').as_posix()}",
            f"/compact what is in @{elsewhere.as_posix()}/",
            "/model claude-opus-4-1 then @note.txt",
        ]
        session = _session()
        for turn, prompt in enumerate(prompts):
            await _run_turn(
                turns, provider, ledger, session, turn, prompt, new_pod=turn == 2
            )

    sent = json.dumps([body for _, body in provider.bodies])
    assert _NOTE not in sent
    assert "unnamed.txt" not in sent
    for turn in range(len(prompts)):
        bodies = provider.bodies_in(turn)
        assert bodies, f"turn {turn} made no model request"
        assert {body["model"] for body in bodies} == {SONNET}
        # Each turn's text and every earlier one: no command cleared or
        # compacted the conversation in between.
        expected = [_as_sent(prompt) for prompt in prompts[: turn + 1]]
        assert _user_texts(bodies[-1])[-len(expected) :] == expected
    sid = session.session_id
    assert turns.resumed == [(0, None), (1, sid), (2, sid)]


@pytest.mark.asyncio(loop_scope="function")
async def test_a_session_stored_earlier_resumes_without_what_its_text_named(
    ledger: PlatformLedger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A CLI session the CLI wrote while it still expanded the text it was
    given holds what it attached for that text. Restored on a new pod and
    resumed, it keeps the turn and drops those attachments."""
    _require_cli()
    async with _RecordingProvider() as provider:
        turns = _StoredEarlierTurns(tmp_path, monkeypatch, provider, _config())
        elsewhere = _write_files(tmp_path)
        note = elsewhere / "note.txt"
        earlier = f"compare @{note.as_posix()} with @{elsewhere.as_posix()}/"
        session = _session()
        await _run_turn(turns, provider, ledger, session, 0, earlier, new_pod=False)
        stored = turns.stored(session.session_id)
        # Whatever reaches the model later can only come from the session.
        note.unlink()
        await _run_turn(turns, provider, ledger, session, 1, "and now?", new_pod=True)

    assert {"file", "directory"} <= _attachment_types(stored)
    assert turns.resumed == [(0, None), (1, session.session_id)]
    later = provider.bodies_in(1)
    assert later, "the resumed turn made no model request"
    assert _user_texts(later[-1])[-2:] == [earlier, "and now?"]
    sent = json.dumps(later)
    assert _NOTE not in sent
    assert "unnamed.txt" not in sent


@pytest.mark.asyncio(loop_scope="function")
async def test_a_session_is_stored_without_what_its_text_named_or_not_at_all(
    ledger: PlatformLedger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The upload leaves out what the CLI attached for the text it was given.
    A session file the upload cannot read whole is not stored, and the copy
    stored before it is removed, so the next turn starts a new CLI session
    from the chat's messages."""
    _require_cli()
    async with _RecordingProvider() as provider:
        turns = _UnreadableSecondTurn(tmp_path, monkeypatch, provider, _config())
        named = _naming_a_file_and_a_directory(_write_files(tmp_path))
        session = _session()
        await _run_turn(turns, provider, ledger, session, 0, named, new_pod=False)
        stored = turns.stored(session.session_id)
        await _run_turn(turns, provider, ledger, session, 1, "and now?", new_pod=False)
        left_stored = turns.has_stored(session.session_id)
        await _run_turn(turns, provider, ledger, session, 2, "and then?", new_pod=False)

    assert {"file", "directory"} <= _attachment_types(turns.written[0])
    assert not {"file", "directory"} & _attachment_types(stored)
    assert _NOTE not in stored.decode()
    assert "unnamed.txt" not in stored.decode()
    assert len(turns.written) == 3
    assert not left_stored
    assert turns.resumed == [(0, None), (1, session.session_id), (2, None)]
    rebuilt = provider.bodies_in(2)
    assert rebuilt, "the turn after the refused session made no model request"
    # One prompt, which carries the chat's earlier turns ahead of its own text.
    [prompt] = _user_texts(rebuilt[-1])
    assert named in prompt and "and now?" in prompt
    assert prompt.endswith("and then?")
    assert _NOTE not in json.dumps(rebuilt)


@pytest.mark.asyncio(loop_scope="function")
async def test_a_stored_copy_that_cannot_be_deleted_stays_until_a_turn_replaces_it(
    ledger: PlatformLedger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When the copy stored before a refused session file cannot be deleted,
    the turn still ends, the copy stays as it was, and the next turn resumes
    from it and stores its own session over it."""
    _require_cli()
    async with _RecordingProvider() as provider:
        turns = _UnreadableSecondTurn(
            tmp_path, monkeypatch, provider, _config(), delete_fails=True
        )
        named = _naming_a_file_and_a_directory(_write_files(tmp_path))
        session = _session()
        await _run_turn(turns, provider, ledger, session, 0, named, new_pod=False)
        stored = turns.stored(session.session_id)
        await _run_turn(turns, provider, ledger, session, 1, "and now?", new_pod=False)
        kept = turns.stored(session.session_id)
        await _run_turn(turns, provider, ledger, session, 2, "and then?", new_pod=False)
        replaced = turns.stored(session.session_id)

    assert kept == stored
    sid = session.session_id
    assert turns.resumed == [(0, None), (1, sid), (2, sid)]
    resumed = provider.bodies_in(2)
    assert resumed, "the turn after the refused session made no model request"
    # Turn 1 is filled in from the chat only when its messages are numbered,
    # which this suite's are not, so only the stored turn and the new one show.
    texts = _user_texts(resumed[-1])
    assert texts[0] == named
    assert texts[-1].endswith("and then?")
    assert b"and then?" in replaced and b"and then?" not in stored


def _require_cli() -> None:
    """The Linux SDK wheel bundles the CLI, so there a missing one fails the
    test instead of skipping it."""
    if _resolve_cli_path() is not None:
        return
    if platform.system() == "Linux":
        pytest.fail("No Claude Code CLI binary: the Linux SDK wheel bundles one")
    pytest.skip("No Claude Code CLI binary: this platform's SDK wheel has none")


class _RecordingProvider(FakeProvider):
    """The fake provider, keeping the body of every Messages request."""

    def __init__(self) -> None:
        super().__init__()
        self.bodies: list[tuple[int, dict[str, Any]]] = []

    def bodies_in(self, turn: int) -> list[dict[str, Any]]:
        return [body for request_turn, body in self.bodies if request_turn == turn]

    async def _messages(self, request: web.Request, route: Route) -> web.StreamResponse:
        self.bodies.append((self.turn, await request.json()))
        return await super()._messages(request, route)


class _ExpandingFirstTurn(SdkTurns):
    """Turn 0 runs with the CLI expanding the text it is given, as it did
    before the copilot sent chat text verbatim."""

    def stored(self, session_id: str) -> bytes:
        content = self._transcripts[session_id].content
        return content if isinstance(content, bytes) else content.encode()

    def has_stored(self, session_id: str) -> bool:
        return session_id in self._transcripts

    def _client(
        self, *args: Any, options: ClaudeAgentOptions, **kwargs: Any
    ) -> ClaudeSDKClient:
        if self._turn == 0:
            options.verbatim_prompts = False
        return super()._client(*args, options=options, **kwargs)


class _StoredEarlierTurns(_ExpandingFirstTurn):
    """Turn 0 is also stored as the CLI wrote it, the way a session written
    before the copilot sent chat text verbatim was kept."""

    def _stubs(self, session: ChatSession) -> dict[str, Any]:
        stubs = super()._stubs(session)
        if self._turn == 0:
            stubs["read_cli_session_from_disk"] = _session_file_as_written
        return stubs


class _UnreadableSecondTurn(_ExpandingFirstTurn):
    """Every turn goes through the production upload reader; turn 1's session
    file first gains a line cut off mid-write. ``written`` keeps each file as
    the CLI left it. With ``delete_fails``, deleting a stored copy fails the
    way ``delete_transcript`` reports a JSONL it could not delete."""

    def __init__(self, *args: Any, delete_fails: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.written: list[bytes] = []
        self._delete_fails = delete_fails

    def _stubs(self, session: ChatSession) -> dict[str, Any]:
        stubs = super()._stubs(session)
        stubs["read_cli_session_from_disk"] = self._read_session
        if self._delete_fails:
            stubs["delete_transcript"] = AsyncMock(
                side_effect=OSError("storage unavailable")
            )
        return stubs

    def _read_session(self, sdk_cwd: str, session_id: str, log_prefix: str):
        path = Path(cli_session_path(sdk_cwd, session_id))
        self.written.append(path.read_bytes())
        if self._turn == 1:
            path.write_bytes(self.written[-1] + _CUT_OFF_LINE)
        return _READ_SESSION(sdk_cwd, session_id, log_prefix)


def _session_file_as_written(sdk_cwd: str, session_id: str, _log_prefix: str) -> bytes:
    return Path(cli_session_path(sdk_cwd, session_id)).read_bytes()


def _config() -> ChatConfig:
    """A direct-Anthropic deployment, pinned so the deployment's own settings
    cannot change the path a turn takes, on the CLI ``_require_cli`` found."""
    cli = _resolve_cli_path()
    return sdk_service.config.model_copy(
        update={
            "use_local": False,
            "use_claude_code_subscription": False,
            "use_e2b_sandbox": False,
            "use_openrouter": False,
            "api_key": "sk-ant-chat-text-e2e",
            "claude_agent_model": SONNET,
            "claude_agent_fallback_model": None,
            "claude_agent_cli_path": str(cli) if cli else None,
            "claude_agent_use_resume": True,
        }
    )


def _session() -> ChatSession:
    session = ChatSession.new("chat-text-e2e-user", dry_run=False)
    # A titled chat skips title generation, a separate model call.
    session.title = "Chat text e2e"
    return session


def _write_files(tmp_path: Path) -> Path:
    """A note in the turn's working directory, and a note and a file nothing
    names in a directory outside it, which is returned."""
    (tmp_path / "copilot-cwd" / "note.txt").write_text(_NOTE)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "note.txt").write_text(_NOTE)
    (elsewhere / "unnamed.txt").write_text(_NOTE)
    return elsewhere


def _naming_a_file_and_a_directory(elsewhere: Path) -> str:
    """Text naming the note outside the working directory, and its directory."""
    return (
        f"compare @{(elsewhere / 'note.txt').as_posix()} "
        f"with @{elsewhere.as_posix()}/"
    )


async def _run_turn(
    turns: SdkTurns,
    provider: FakeProvider,
    ledger: PlatformLedger,
    session: ChatSession,
    turn: int,
    prompt: str,
    *,
    new_pod: bool,
) -> None:
    provider.turn = ledger.turn = turn
    session.messages.append(ChatMessage(role="user", content=prompt))
    await turns(session, prompt, new_pod=new_pod)
    await ledger.settle()


def _as_sent(prompt: str) -> str:
    """The copilot puts a space before a leading "/", so the CLI cannot read
    the text as one of its commands."""
    return f" {prompt}" if prompt.startswith("/") else prompt


def _user_texts(body: dict[str, Any]) -> list[str]:
    """Each user message's own text in a request, without the reminders the
    CLI adds."""
    texts: list[str] = []
    for message in body["messages"]:
        if message["role"] != "user":
            continue
        content = message["content"]
        blocks = (
            [{"type": "text", "text": content}] if isinstance(content, str) else content
        )
        texts += [
            block["text"]
            for block in blocks
            if block.get("type") == "text"
            and not block["text"].startswith("<system-reminder>")
        ]
    return texts


def _attachment_types(session: bytes) -> set[str]:
    """The types of the attachment entries in a CLI session file."""
    entries = [json.loads(line) for line in session.decode().splitlines() if line]
    return {
        entry["attachment"].get("type", "")
        for entry in entries
        if entry.get("type") == "attachment"
    }
