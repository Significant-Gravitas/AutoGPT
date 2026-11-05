"""A computer-use call that reaches the disk runs only once the user approves it.

End to end through the real gate, its card and held call, the SDK's MCP
wrapper and the computer-use handler, with the review table, the held-call
store and the shim faked. The input is Otto's call; the output is what the
shim is asked to run, and what Otto reads back.
"""

import base64
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus

import backend.copilot.sdk.computer_use_tools as tools
from backend.copilot.gate import held
from backend.copilot.gate import review as review_store
from backend.copilot.gate.classifier import Judgement
from backend.copilot.model import (
    AutopilotMode,
    ChatMessage,
    ChatSession,
    ChatSessionMetadata,
)
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.sdk import tool_adapter
from backend.copilot.sdk.computer_use_approval import computer_subject
from backend.copilot.tools.local_pc_shim import ComputerUseDriver, LocalPCShim
from backend.copilot.trial_cost_context import TrialCostContext
from backend.util.json import sanitize_json

_GATE = "backend.copilot.gate"
_TOOLS = ["click", "clipboard_write", "get_window_state", "launch_app", "run_script"]
_CATALOG = {
    "driver": {"name": "cua-driver", "version": "0.38.0", "runtime": "direct"},
    "instructions": "cua-driver: background GUI automation.",
    "tools": [
        {
            "name": "clipboard_write",
            "description": "Replace the clipboard.",
            "input_schema": {"type": "object", "properties": {"file_path": {}}},
            "approval_arguments": {
                "file_path": "puts this file on the clipboard, to paste into an app"
            },
        },
        {
            "name": "get_window_state",
            "description": "Read a window.",
            "input_schema": {
                "type": "object",
                "properties": {"screenshot_out_file": {}},
            },
            "approval_arguments": {
                "screenshot_out_file": "writes the screenshot to this file"
            },
        },
        {
            "name": "click",
            "description": "Click an element.",
            "input_schema": {"type": "object", "properties": {"element_token": {}}},
        },
        {
            "name": "run_script",
            "description": "Run a JavaScript program that drives apps.",
            "input_schema": {"type": "object", "properties": {"script": {}}},
            "approval_arguments": {
                "script": "runs this script, which drives apps and can save a "
                "screenshot to any file"
            },
        },
        {
            "name": "launch_app",
            "description": "Launch an app.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {},
                    "launch_path": {},
                    "additional_arguments": {},
                    "urls": {},
                },
            },
            "approval_arguments": {
                "launch_path": "runs this command line",
                "additional_arguments": (
                    "starts the app with these command-line arguments"
                ),
                "urls": "opens these links or files, each in the app that handles it",
            },
        },
    ],
}
_FILE = {"tool": "clipboard_write", "arguments": {"file_path": "/home/me/cv.pdf"}}
_CLICK = {"tool": "click", "arguments": {"element_token": "s1:4"}}


class _Rows:
    """The review table and the held-call store, as far as the gate uses them."""

    def __init__(self):
        self.rows: dict[str, SimpleNamespace] = {}
        self.held: list[held.HeldCall] = []

    async def find_review(self, review_id, user_id, session_id):
        return self.rows.get(review_id)

    async def consume(self, review_id, user_id):
        return self.rows.pop(review_id, None) is not None

    async def remember(self, session_id, call):
        self.held.append(call)
        return True

    async def held_calls(self, session_id):
        return {call.review_id: call for call in self.held}

    async def claim(self, session_id, review_id):
        kept = [call for call in self.held if call.review_id != review_id]
        claimed, self.held = len(kept) < len(self.held), kept
        return claimed

    async def get_reviews_by_node_exec_ids(self, ids, user_id):
        return {i: self.rows[i] for i in ids if i in self.rows}

    async def open_review_row(self, review_id, user_id, session, payload, message):
        self.rows[review_id] = SimpleNamespace(
            node_exec_id=review_id,
            session_id=session.session_id,
            status=ReviewStatus.WAITING,
            created_at=datetime.now(UTC),
            updated_at=None,
            reviewed_at=None,
            payload=sanitize_json(payload),
        )
        return True

    def answer(self, status: ReviewStatus) -> SimpleNamespace:
        (row,) = self.rows.values()
        row.status = status
        row.reviewed_at = datetime.now(UTC)
        return row


@pytest.fixture(autouse=True)
def _deployment_allows_computer_use(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both settings are off by default; a deployment turns them on."""
    monkeypatch.setattr(tools._chat_config, "use_local_pc_executor", True)
    monkeypatch.setattr(tools._chat_config, "allow_computer_use", True)


@pytest.fixture
def supervisor() -> AsyncMock:
    """A supervisor that would let anything run: only the user's approval may
    stand for a call that reaches the disk."""
    return AsyncMock(return_value=Judgement(allowed=True, reason="looks fine"))


@pytest.fixture
def rows(supervisor: AsyncMock):
    fake = _Rows()
    with (
        patch(f"{_GATE}.is_feature_enabled", AsyncMock(return_value=True)),
        patch(f"{_GATE}.supervise", supervisor),
        patch(
            f"{_GATE}.capture_cost_context",
            AsyncMock(return_value=TrialCostContext(user_id="user-1", trial_id=None)),
        ),
        patch(f"{_GATE}.chat_rules.rule_for", AsyncMock(return_value=None)),
        patch(f"{_GATE}.chat_rules.set_ask", AsyncMock()),
        patch.object(review_store, "find_review", fake.find_review),
        patch.object(review_store, "consume", fake.consume),
        patch.object(review_store, "open_review_row", fake.open_review_row),
        patch.object(held, "remember", fake.remember),
        patch.object(held, "_held", fake.held_calls),
        patch.object(held, "_claim", fake.claim),
        patch.object(held, "review_db", lambda: fake),
    ):
        yield fake


@pytest.fixture
def computer(monkeypatch: pytest.MonkeyPatch) -> LocalPCShim:
    """A connected shim offering computer use, consented to in this chat; the
    read judge passes every result through."""
    shim = LocalPCShim.__new__(LocalPCShim)
    shim.sandbox_id = "session-1"
    shim.platform = "linux"
    shim.machine_id = "machine-1"
    shim.allowed_root = "/home/me/workspace"
    shim.capabilities = ["computer_use"]
    shim.capability_set = frozenset(shim.capabilities)
    shim.computer_use_driver = ComputerUseDriver(
        name="cua-driver", version="0.38.0", runtime="direct"
    )
    shim.computer_use_tools = list(_TOOLS)
    shim._connection_generation = 1
    shim.computer = MagicMock()
    shim.computer.tools = AsyncMock(return_value=_CATALOG)
    shim.computer.call = AsyncMock(
        return_value={
            "tool": "clipboard_write",
            "is_error": False,
            "content": [{"type": "text", "text": "Copied."}],
        }
    )
    monkeypatch.setattr(tools, "is_computer_use_approved", AsyncMock(return_value=True))
    monkeypatch.setattr(
        tool_adapter, "release_non_registry_read", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        tool_adapter,
        "screen_non_registry_read",
        AsyncMock(side_effect=lambda *args, **_kwargs: args[4]),
    )
    return shim


def _session(mode: AutopilotMode) -> ChatSession:
    return ChatSession(
        session_id="session-1",
        user_id="user-1",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=ChatSessionMetadata(origin="interactive", autopilot_mode=mode),
        messages=[ChatMessage(role="user", content="attach my CV to the form")],
    )


def _offered(computer_use: bool = True) -> dict[str, held.McpHandler]:
    """The replay table a turn builds, with computer use on or off for it."""
    return tool_adapter.held_call_handlers(
        use_local_pc_computer=computer_use, use_recording=False
    )


def _call_as_otto(session: ChatSession, shim: LocalPCShim):
    tool_adapter.set_execution_context("user-1", session, sandbox=shim)
    return _offered()[tools.LOCAL_PC_COMPUTER_TOOL]


async def _next_turn(
    session: ChatSession, *, computer_use: bool = True
) -> tuple[list[str], list[PendingMessage]]:
    delivered: list[PendingMessage] = []
    heard = [
        status.message
        async for status in held.resolve_answered(
            "user-1",
            session,
            delivered.append,
            mcp_handlers=_offered(computer_use),
        )
    ]
    return heard, delivered


def _sent(shim: LocalPCShim) -> list[tuple[str, list[str]]]:
    return [
        (call.args[0], call.kwargs["approved_arguments"])
        for call in shim.computer.call.await_args_list
    ]


@pytest.mark.parametrize("mode", ["auto", "ask_first"])
async def test_a_disk_call_runs_once_the_user_approves_its_card(
    rows, computer, supervisor, mode: AutopilotMode
):
    session = _session(mode)
    call = _call_as_otto(session, computer)

    parked = await call(_FILE)

    assert json.loads(parked["content"][0]["text"])["type"] == "approval_required"
    assert _sent(computer) == []
    supervisor.assert_not_awaited()
    (card,) = rows.rows.values()
    headline = card.payload["headline"]
    assert headline["ask"] == "Copy a file on your computer to its clipboard"
    assert headline["object"] == "/home/me/cv.pdf"

    rows.answer(ReviewStatus.APPROVED)
    heard, delivered = await _next_turn(session)

    assert _sent(computer) == [("clipboard_write", ["file_path"])]
    assert heard == [
        "Running the action you approved: Copy a file on your computer to its "
        "clipboard “/home/me/cv.pdf”"
    ]
    (late,) = delivered
    assert late.metadata["held_call"]["outcome"] == "approved"
    assert "Copied." in late.content
    # The approval is spent, so the same call asks again.
    assert rows.rows == {}
    await call(_FILE)
    assert len(rows.rows) == 1 and len(_sent(computer)) == 1


async def test_a_declined_disk_call_never_runs(rows, computer):
    session = _session("auto")
    await _call_as_otto(session, computer)(_FILE)

    rows.answer(ReviewStatus.REJECTED)
    _, delivered = await _next_turn(session)

    assert _sent(computer) == []
    (late,) = delivered
    assert late.metadata["held_call"]["outcome"] == "rejected"


async def test_an_unsupervised_chat_runs_a_disk_call_without_asking(rows, computer):
    result = await _call_as_otto(_session("unsupervised"), computer)(_FILE)

    assert result["isError"] is False
    assert _sent(computer) == [("clipboard_write", ["file_path"])]
    assert rows.rows == {}


async def test_where_the_gate_cannot_ask_a_disk_call_is_refused(rows, computer):
    with patch(f"{_GATE}.is_feature_enabled", AsyncMock(return_value=False)):
        result = await _call_as_otto(_session("auto"), computer)(_FILE)

    assert json.loads(result["content"][0]["text"])["code"] == "APPROVAL_REQUIRED"
    assert _sent(computer) == []
    assert rows.rows == {}


async def test_a_tool_list_that_changes_under_the_call_approves_nothing(rows, computer):
    """The gate decided on the list it read; the handler acts on the one the
    shim has now. A disk argument neither the user nor the mode ruled on stays
    refused."""
    before = {
        **_CATALOG,
        "tools": [
            {k: v for k, v in tool.items() if k != "approval_arguments"}
            for tool in _CATALOG["tools"]
        ],
    }
    computer.computer.tools = AsyncMock(side_effect=[before, _CATALOG])

    result = await _call_as_otto(_session("unsupervised"), computer)(_FILE)

    assert json.loads(result["content"][0]["text"])["code"] == "APPROVAL_REQUIRED"
    assert _sent(computer) == []


async def test_an_ordinary_call_in_auto_is_left_to_the_supervisor(
    rows, computer, supervisor
):
    result = await _call_as_otto(_session("auto"), computer)(_CLICK)

    assert result["isError"] is False
    supervisor.assert_awaited_once()
    assert _sent(computer) == [("click", [])]


async def test_a_screenshot_saved_to_disk_warns_it_cannot_be_undone(rows, computer):
    session = _session("auto")
    shot = {
        "tool": "get_window_state",
        "arguments": {"pid": 4, "screenshot_out_file": "/home/me/shot.png"},
    }

    await _call_as_otto(session, computer)(shot)

    (card,) = rows.rows.values()
    assert card.payload["headline"]["ask"] == "Save a screenshot on your computer"
    assert card.payload["subject"]["irreversible"] is True


@pytest.mark.parametrize("mode", ["auto", "ask_first"])
async def test_a_script_runs_once_the_user_approves_it(
    rows, computer, supervisor, mode: AutopilotMode
):
    """The shim can't see a script's calls, and one can save a screenshot to
    any file, so a script asks like a call that names a file, even in Auto,
    where an ordinary call is left to the supervisor."""
    session = _session(mode)
    script = 'const app = await cua.getApp("Calculator");\n  return app.query({})'
    run = {"tool": "run_script", "arguments": {"script": script}}

    parked = await _call_as_otto(session, computer)(run)

    assert json.loads(parked["content"][0]["text"])["type"] == "approval_required"
    assert _sent(computer) == []
    supervisor.assert_not_awaited()
    (card,) = rows.rows.values()
    headline = card.payload["headline"]
    assert headline["ask"] == "Run a script on your computer"
    assert (
        headline["object"]
        == 'const app = await cua.getApp("Calculator"); return app.query({})'
    )
    assert card.payload["subject"]["irreversible"] is False

    rows.answer(ReviewStatus.APPROVED)
    await _next_turn(session)

    assert _sent(computer) == [("run_script", ["script"])]


@pytest.mark.parametrize("mode", ["auto", "ask_first"])
async def test_a_script_too_long_for_its_card_is_refused_before_it_asks(
    rows, computer, mode: AutopilotMode
):
    """cua-driver takes a 64 KiB script, but a card shows about 26,000
    characters of a call, so a benign head could carry an unseen tail that
    ran on the user's approval of the head."""
    session = _session(mode)
    script = "return 1;\n" + "// padding\n" * 3_000 + 'cua.call("launch_app", {})'

    refused = await _call_as_otto(session, computer)(
        {"tool": "run_script", "arguments": {"script": script}}
    )

    assert refused["isError"] is True
    assert json.loads(refused["content"][0]["text"])["code"] == "TOO_LONG_TO_APPROVE"
    assert rows.rows == {}
    assert _sent(computer) == []


async def test_a_script_that_fits_is_shown_whole_on_its_card(rows, computer):
    session = _session("auto")
    script = "return 1;\n" + "// padding\n" * 2_000

    await _call_as_otto(session, computer)(
        {"tool": "run_script", "arguments": {"script": script}}
    )

    (card,) = rows.rows.values()
    assert card.payload["clipped"] == []
    assert card.payload["arguments"]["arguments"]["script"] == script


async def test_a_launch_that_runs_a_command_line_asks_even_in_auto(
    rows, computer, supervisor
):
    """cua-driver starts a launch's command line or extra arguments out of
    sight, with no window of its own, and a Chromium flag such as
    --renderer-cmd-prefix runs any command, so a launch ran anything on the
    supervisor's say in Auto."""
    session = _session("auto")
    launch = {
        "tool": "launch_app",
        "arguments": {
            "name": "Google Chrome",
            "additional_arguments": ["--renderer-cmd-prefix=sh -c id"],
        },
    }

    parked = await _call_as_otto(session, computer)(launch)

    assert json.loads(parked["content"][0]["text"])["type"] == "approval_required"
    assert _sent(computer) == []
    supervisor.assert_not_awaited()
    (card,) = rows.rows.values()
    headline = card.payload["headline"]
    assert headline["ask"] == (
        "Start an app on your computer with command-line arguments"
    )
    assert headline["object"] == "Google Chrome, --renderer-cmd-prefix=sh -c id"

    rows.answer(ReviewStatus.APPROVED)
    await _next_turn(session)

    assert _sent(computer) == [("launch_app", ["additional_arguments"])]


async def test_opening_an_app_by_name_is_left_to_the_supervisor(
    rows, computer, supervisor
):
    opening = {"tool": "launch_app", "arguments": {"name": "Calculator"}}

    result = await _call_as_otto(_session("auto"), computer)(opening)

    assert result["isError"] is False
    supervisor.assert_awaited_once()
    assert _sent(computer) == [("launch_app", [])]


def test_a_launch_is_asked_about_by_what_it_runs():
    """The ask came from the alphabetically first argument, so a command line
    with extra arguments read as "start an app with arguments"."""
    spec = next(t for t in _CATALOG["tools"] if t["name"] == "launch_app")

    subject = computer_subject(
        "launch_app",
        {"launch_path": "sh -c id", "additional_arguments": ["-x"]},
        spec,
    )

    assert subject.name == "Run a program on your computer"
    assert subject.reason.startswith("`launch_app` runs this command line.")
    assert subject.preview is not None
    assert subject.preview.object == "sh -c id, -x"


async def test_an_approved_screenshot_call_brings_its_image_back(rows, computer):
    session = _session("ask_first")
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 32).decode()
    computer.computer.call = AsyncMock(
        return_value={
            "tool": "get_window_state",
            "is_error": False,
            "content": [
                {"type": "text", "text": "Saved the window."},
                {"type": "image", "data": png, "mime_type": "image/png"},
            ],
        }
    )
    shot = {
        "tool": "get_window_state",
        "arguments": {"pid": 4, "screenshot_out_file": "/home/me/shot.png"},
    }
    await _call_as_otto(session, computer)(shot)

    rows.answer(ReviewStatus.APPROVED)
    _, delivered = await _next_turn(session)

    assert _sent(computer) == [("get_window_state", ["screenshot_out_file"])]
    (late,) = delivered
    assert "Saved the window." in late.content
    assert late.model_image is not None
    assert late.model_image.data_base64 == png


async def test_an_approved_script_brings_back_its_last_image_and_names_the_rest(
    rows, computer
):
    """cua-driver's run_script returns its last two images, a before and an
    after. The replay kept the first, so Otto saw the screen as it was
    before the script ran, and nothing said another image existed."""
    session = _session("ask_first")
    before = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"b" * 32).decode()
    after = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"a" * 32).decode()
    computer.computer.call = AsyncMock(
        return_value={
            "tool": "run_script",
            "is_error": False,
            "content": [
                {"type": "image", "data": before, "mime_type": "image/png"},
                {"type": "text", "text": "Clicked Save."},
                {"type": "image", "data": after, "mime_type": "image/png"},
            ],
        }
    )
    await _call_as_otto(session, computer)(
        {"tool": "run_script", "arguments": {"script": "return 1"}}
    )

    rows.answer(ReviewStatus.APPROVED)
    _, delivered = await _next_turn(session)

    (late,) = delivered
    assert late.model_image is not None
    assert late.model_image.data_base64 == after
    assert "Clicked Save." in late.content
    assert "1 more image(s) from this call aren't shown" in late.content


async def test_a_held_call_does_not_run_once_computer_use_is_off(rows, computer):
    """The replay table held every computer-use handler, so a call approved
    after the deployment or the chat stopped offering computer use still
    drove the computer on the next turn."""
    session = _session("ask_first")
    await _call_as_otto(session, computer)(_CLICK)
    rows.answer(ReviewStatus.APPROVED)

    _heard, delivered = await _next_turn(session, computer_use=False)

    assert _sent(computer) == []
    (late,) = delivered
    assert late.metadata["held_call"]["outcome"] == "closed"
    assert "Nothing ran" in late.content
    # The approval is spent: it can't run on a later turn either.
    _heard, again = await _next_turn(session)
    assert again == [] and _sent(computer) == []


async def test_an_ordinary_call_held_in_ask_first_runs_after_approval(rows, computer):
    session = _session("ask_first")
    call = _call_as_otto(session, computer)

    await call(_CLICK)
    (card,) = rows.rows.values()
    assert card.payload["headline"]["ask"] == "Use your computer"
    assert card.payload["headline"]["object"] == "click"

    rows.answer(ReviewStatus.APPROVED)
    _, delivered = await _next_turn(session)

    assert _sent(computer) == [("click", [])]
    (late,) = delivered
    assert late.metadata["held_call"]["outcome"] == "approved"


async def test_reading_the_tool_list_never_asks(rows, computer):
    tool_adapter.set_execution_context(
        "user-1", _session("ask_first"), sandbox=computer
    )
    read = _offered()[tools.LOCAL_PC_COMPUTER_TOOLS_TOOL]

    result = await read({"tools": ["clipboard_write"]})

    assert result["isError"] is False
    assert rows.rows == {}
    listed = json.loads(result["content"][0]["text"])
    assert {
        tool["name"]: tool.get("approval_arguments") for tool in listed["tools"]
    } == {
        "clipboard_write": {
            "file_path": "puts this file on the clipboard, to paste into an app"
        },
        "get_window_state": {
            "screenshot_out_file": "writes the screenshot to this file"
        },
        "click": None,
        "run_script": {
            "script": "runs this script, which drives apps and can save a "
            "screenshot to any file"
        },
        "launch_app": {
            "launch_path": "runs this command line",
            "additional_arguments": "starts the app with these command-line arguments",
            "urls": "opens these links or files, each in the app that handles it",
        },
    }


_UPLOAD = {
    "name": "browser_set_input_files",
    "approval_arguments": {"files": "hands these files to the page"},
}


def test_every_file_handed_to_a_page_is_named_on_the_card():
    subject = computer_subject(
        "browser_set_input_files",
        {"ref": "p1:2", "files": ["/home/me/cv.pdf", "/home/me/letter.pdf"]},
        _UPLOAD,
    )

    assert subject.preview is not None
    assert subject.preview.ask == "Give a web page files from your computer"
    assert subject.preview.object == "/home/me/cv.pdf, /home/me/letter.pdf"
    assert subject.irreversible is False


def test_a_long_list_of_files_is_cut_short_on_the_card():
    files = [f"/home/me/photos/{n:04}.jpg" for n in range(50)]

    subject = computer_subject("browser_set_input_files", {"files": files}, _UPLOAD)

    assert subject.preview is not None and subject.preview.object is not None
    assert len(subject.preview.object) == 300
    assert subject.preview.object.endswith("…")


def test_an_empty_disk_argument_is_an_ordinary_call():
    spec = {"approval_arguments": {"screenshot_out_file": "writes it"}}

    subject = computer_subject(
        "get_window_state", {"pid": 1, "screenshot_out_file": ""}, spec
    )

    assert subject.key == "local_pc_computer"
