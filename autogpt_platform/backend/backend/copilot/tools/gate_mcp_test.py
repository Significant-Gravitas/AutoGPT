"""The gate decides an MCP call on the server's effect map, and asks on first
use of any tool the map does not name.

Driven through ``BaseTool.execute`` so the subject hook, the gate, the chat
rules and the MCP path are the ones the engines call.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.gate import chat_rules
from backend.copilot.gate.review import payload_headline, review_payload
from backend.copilot.model import AutopilotMode, ChatSession, ChatSessionMetadata
from backend.copilot.tools.models import MCPToolOutputResponse
from backend.copilot.tools.run_capability import RunCapabilityTool

_GATE = "backend.copilot.gate"
_CAP = "backend.copilot.tools.run_capability"
_GITHUB = "mcp:api.githubcopilot.com"
_OPEN_WORLD = "https://mcp.example.com/mcp"


def _session(mode: AutopilotMode = "auto") -> ChatSession:
    return ChatSession(
        session_id="session-1",
        user_id="user-1",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=ChatSessionMetadata(origin="interactive", autopilot_mode=mode),
        messages=[],
    )


class _Redis:
    """Chat rules are plain keys; the real module builds and reads them."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.data[key] = value

    async def get(self, key: str) -> str | None:
        return self.data.get(key)


@pytest.fixture
def gate():
    """The gate on, no approval on record, chat rules on a fresh store."""
    store = SimpleNamespace(
        find_review=AsyncMock(return_value=None),
        open_review=AsyncMock(return_value=True),
        consume=AsyncMock(return_value=True),
        own_review=AsyncMock(return_value="copilot-mcp-x:1"),
        classify=AsyncMock(return_value=(True, "")),
    )
    with (
        patch(f"{_GATE}.is_feature_enabled", AsyncMock(return_value=True)),
        patch(f"{_GATE}.review_store.find_review", store.find_review),
        patch(f"{_GATE}.review_store.open_review", store.open_review),
        patch(f"{_GATE}.review_store.consume", store.consume),
        patch(f"{_GATE}.held.remember", AsyncMock(return_value=True)),
        patch(f"{_GATE}.classify", store.classify),
        patch(f"{_GATE}.reads.release_held_read", AsyncMock(return_value=None)),
        patch(f"{_GATE}.reads.screen_read", AsyncMock(return_value=None)),
        patch(f"{_CAP}.open_mcp_review", store.own_review),
        patch(
            "backend.copilot.gate.chat_rules.get_redis_async",
            AsyncMock(return_value=_Redis()),
        ),
    ):
        yield store


@pytest.fixture
def ran():
    """What reached the MCP server, if anything did."""
    run = AsyncMock(
        return_value=MCPToolOutputResponse(
            message="ran", server_url="u", tool_name="t", session_id="session-1"
        )
    )
    with patch(f"{_CAP}.RunMCPToolTool._execute", run):
        yield run


async def _call(
    session: ChatSession,
    server: str,
    tool: str,
    arguments: dict[str, Any] | None = None,
):
    return await RunCapabilityTool().execute(
        "user-1",
        session,
        "call-1",
        id=server,
        input={"tool": tool, "arguments": arguments or {}},
    )


def _is_held(result) -> bool:
    return "approval_required" in str(result.output)


def _headline(gate) -> str:
    _, _, _, tool_name, args, reason, subject = gate.open_review.await_args.args
    card = payload_headline(review_payload(tool_name, args, subject))
    assert card == f"Run “{subject.name}”"
    return f"{subject.name} — {reason}"


@pytest.mark.parametrize("mode", ["ask_first", "auto"])
async def test_a_mapped_read_runs_and_a_mapped_write_asks(gate, ran, mode):
    read = await _call(_session(mode), _GITHUB, "issue_read", {"issue_number": 1})
    assert not _is_held(read)
    ran.assert_awaited_once()

    write = await _call(_session(mode), _GITHUB, "create_branch", {"branch": "x"})
    assert _is_held(write)
    ran.assert_awaited_once()
    assert _headline(gate) == (
        "create_branch on api.githubcopilot.com — reaches outside the platform"
    )


async def test_an_unmapped_tool_asks_naming_the_host_and_the_tool(gate, ran):
    result = await _call(_session(), _GITHUB, "brand_new_tool")
    assert _is_held(result)
    ran.assert_not_awaited()
    assert _headline(gate) == (
        "brand_new_tool on api.githubcopilot.com — its effect is unknown"
    )


@pytest.mark.parametrize("mode", ["ask_first", "auto"])
async def test_a_read_shaped_name_on_an_unknown_server_still_asks(gate, ran, mode):
    """A tool's name is no evidence: the verb heuristic never runs under the flag."""
    result = await _call(_session(mode), _OPEN_WORLD, "get_things")
    assert _is_held(result)
    ran.assert_not_awaited()
    assert "get_things on mcp.example.com" in _headline(gate)


async def test_a_chat_allow_covers_that_tool_and_no_other_on_the_host(gate, ran):
    session = _session()
    assert _is_held(await _call(session, _OPEN_WORLD, "do_thing", {"n": 1}))
    await _answer_with_rule(gate, "allow")

    assert not _is_held(await _call(session, _OPEN_WORLD, "do_thing", {"n": 2}))
    ran.assert_awaited_once()
    assert _is_held(await _call(session, _OPEN_WORLD, "other_thing"))
    ran.assert_awaited_once()


async def test_a_chat_judge_sends_the_next_call_to_the_supervisor(gate, ran):
    session = _session("ask_first")
    assert _is_held(await _call(session, _OPEN_WORLD, "do_thing", {"n": 1}))
    gate.classify.assert_not_awaited()
    await _answer_with_rule(gate, "judge")

    assert not _is_held(await _call(session, _OPEN_WORLD, "do_thing", {"n": 2}))
    gate.classify.assert_awaited_once()
    ran.assert_awaited_once()


async def test_a_chat_allow_does_not_reach_another_server_on_the_host(gate, ran):
    session = _session()
    server_a, server_b = (
        "https://tools.example.com/a/mcp",
        "https://tools.example.com/b/mcp",
    )
    assert _is_held(await _call(session, server_a, "search", {"q": 1}))
    await _answer_with_rule(gate, "allow")

    assert not _is_held(await _call(session, server_a, "search", {"q": 2}))
    assert _is_held(await _call(session, server_b, "search", {"q": 2}))
    ran.assert_awaited_once()


async def test_a_chat_judge_covers_an_irreversible_tool_too(gate, ran):
    """Judging means the supervisor decides next time, irreversible or not."""
    session = _session()
    assert _is_held(await _call(session, _GITHUB, "merge_pull_request", {"n": 1}))
    await _answer_with_rule(gate, "judge")

    assert not _is_held(await _call(session, _GITHUB, "merge_pull_request", {"n": 2}))
    gate.classify.assert_awaited_once()
    ran.assert_awaited_once()


async def test_an_irreversible_tool_asks_once_on_the_gate_row_only(gate, ran):
    result = await _call(_session(), _GITHUB, "merge_pull_request", {"n": 7})
    assert _is_held(result)
    assert _headline(gate).endswith("— cannot be taken back")
    gate.open_review.assert_awaited_once()

    gate.find_review.return_value = SimpleNamespace(status=ReviewStatus.APPROVED)
    approved = await _call(_session(), _GITHUB, "merge_pull_request", {"n": 7})
    assert not _is_held(approved)
    ran.assert_awaited_once()
    gate.own_review.assert_not_awaited()


async def test_an_approved_first_use_runs_without_a_second_card(gate, ran):
    """The server-side review would ask again for an uncatalogued server."""
    gate.find_review.return_value = SimpleNamespace(status=ReviewStatus.APPROVED)
    result = await _call(_session(), _OPEN_WORLD, "do_thing")
    assert not _is_held(result)
    ran.assert_awaited_once()
    gate.own_review.assert_not_awaited()


async def test_unsupervised_runs_a_first_use_without_asking(gate, ran):
    result = await _call(_session("unsupervised"), _OPEN_WORLD, "do_thing")
    assert not _is_held(result)
    ran.assert_awaited_once()


async def test_listing_a_servers_tools_never_asks(gate, ran):
    result = await RunCapabilityTool().execute(
        "user-1", _session("ask_first"), "call-1", id=_OPEN_WORLD, input={}
    )
    assert not _is_held(result)
    gate.open_review.assert_not_awaited()


async def _answer_with_rule(gate, rule: chat_rules.ChatRule) -> None:
    """What the approve endpoint does for the card the gate just opened."""
    subject = gate.open_review.await_args.args[6]
    row = SimpleNamespace(status=ReviewStatus.APPROVED)
    await chat_rules.set_answer_rules(
        "session-1", {"r": row}, {"r": rule}, {"r": subject.key}
    )


@pytest.mark.parametrize(
    "server, tool, pauses",
    [
        (_OPEN_WORLD, "get_things", False),
        (_OPEN_WORLD, "delete_things", True),
        # Catalogued servers run, mapped write or not.
        (_GITHUB, "create_branch", False),
    ],
)
async def test_flag_off_the_verb_heuristic_decides_as_before(ran, server, tool, pauses):
    own_review = AsyncMock(return_value="copilot-mcp-x:1")
    with (
        patch(f"{_GATE}.is_feature_enabled", AsyncMock(return_value=False)),
        patch(f"{_CAP}.open_mcp_review", own_review),
        patch(f"{_GATE}.reads.screen_read", AsyncMock(return_value=None)),
    ):
        await _call(_session(), server, tool)
    assert own_review.await_count == int(pauses)
    assert ran.await_count == int(not pauses)
