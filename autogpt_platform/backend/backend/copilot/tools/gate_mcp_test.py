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
from backend.copilot.gate.classifier import Judgement
from backend.copilot.gate.headline import Headline
from backend.copilot.gate.review import payload_headline, review_payload
from backend.copilot.model import AutopilotMode, ChatSession, ChatSessionMetadata
from backend.copilot.tools.models import MCPToolOutputResponse
from backend.copilot.tools.run_capability import RunCapabilityTool

_GATE = "backend.copilot.gate"
_CAP = "backend.copilot.tools.run_capability"
_GITHUB = "mcp:api.githubcopilot.com"
_OPEN_WORLD = "https://mcp.example.com/mcp"


def _session(
    mode: AutopilotMode = "auto",
    session_id: str = "session-1",
    expert_id: str | None = None,
) -> ChatSession:
    return ChatSession(
        session_id=session_id,
        expert_id=expert_id,
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
        open_review=AsyncMock(side_effect=_opened),
        consume=AsyncMock(return_value=True),
        own_review=AsyncMock(return_value="copilot-mcp-x:1"),
        classify=AsyncMock(return_value=Judgement(allowed=True, reason="")),
    )
    with (
        patch(f"{_GATE}.is_feature_enabled", AsyncMock(return_value=True)),
        patch(f"{_GATE}.review_store.find_review", store.find_review),
        patch(f"{_GATE}.review_store.open_review", store.open_review),
        patch(f"{_GATE}.review_store.consume", store.consume),
        patch(f"{_GATE}.held.remember", AsyncMock(return_value=True)),
        patch(f"{_GATE}.supervise", store.classify),
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
        "create_branch on api.githubcopilot.com — Runs create_branch on "
        "api.githubcopilot.com, which reaches outside the platform."
    )


async def test_an_unmapped_tool_asks_naming_the_host_and_the_tool(gate, ran):
    result = await _call(_session(), _GITHUB, "brand_new_tool")
    assert _is_held(result)
    ran.assert_not_awaited()
    assert _headline(gate) == (
        "brand_new_tool on api.githubcopilot.com — Otto does not know what "
        "brand_new_tool on api.githubcopilot.com does, so he asks."
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


@pytest.mark.parametrize(
    "scope, same_expert_runs, other_expert_runs",
    [("chat", False, False), ("expert", True, False), ("team", True, True)],
)
async def test_a_rule_reaches_the_chats_its_scope_names(
    gate, ran, scope, same_expert_runs, other_expert_runs
):
    maria = _session(session_id="chat-a", expert_id="maria")
    assert _is_held(await _call(maria, _OPEN_WORLD, "do_thing", {"n": 1}))
    await _answer_with_rule(gate, "allow", maria, scope)

    maria_again = _session(session_id="chat-a2", expert_id="maria")
    max_ = _session(session_id="chat-b", expert_id="max")
    held = _is_held(await _call(maria_again, _OPEN_WORLD, "do_thing", {"n": 2}))
    assert held != same_expert_runs
    held = _is_held(await _call(max_, _OPEN_WORLD, "do_thing", {"n": 3}))
    assert held != other_expert_runs


async def test_an_expert_rule_covers_otto_only_from_an_otto_chat(gate, ran):
    otto = _session(session_id="chat-o")
    assert _is_held(await _call(otto, _OPEN_WORLD, "do_thing", {"n": 1}))
    await _answer_with_rule(gate, "allow", otto, "expert")

    assert not _is_held(
        await _call(_session(session_id="chat-o2"), _OPEN_WORLD, "do_thing", {"n": 2})
    )
    maria = _session(session_id="chat-a", expert_id="maria")
    assert _is_held(await _call(maria, _OPEN_WORLD, "do_thing", {"n": 3}))


@pytest.mark.parametrize("wide", ["expert", "team"])
async def test_a_chat_ask_beats_a_wider_allow(gate, ran, wide):
    maria = _session(session_id="chat-a", expert_id="maria")
    assert _is_held(await _call(maria, _OPEN_WORLD, "do_thing", {"n": 1}))
    subject = gate.open_review.await_args.args[6]
    await _answer_with_rule(gate, "allow", maria, wide)
    maria_again = _session(session_id="chat-a2", expert_id="maria")
    await chat_rules.set_rule("chat-a2", subject.key, "ask")

    assert _is_held(await _call(maria_again, _OPEN_WORLD, "do_thing", {"n": 2}))
    assert not _is_held(await _call(maria, _OPEN_WORLD, "do_thing", {"n": 3}))


async def test_a_rejection_revokes_the_expert_and_team_rules_that_ran_it(gate, ran):
    max_ = _session(session_id="chat-b", expert_id="max")
    assert _is_held(await _call(max_, _OPEN_WORLD, "do_thing", {"n": 1}))
    subject = gate.open_review.await_args.args[6]
    await _answer_with_rule(gate, "allow", max_, "expert")
    await chat_rules.set_scoped_rule("team", "user-1", None, subject.key, "allow")

    max_again = _session(session_id="chat-b2", expert_id="max")
    gate.find_review.return_value = SimpleNamespace(status=ReviewStatus.REJECTED)
    with patch(f"{_GATE}.held.rule_key", AsyncMock(return_value=subject.key)):
        assert _is_held(await _call(max_again, _OPEN_WORLD, "do_thing", {"n": 2}))
    gate.find_review.return_value = None

    for chat, reason in [
        (
            _session(session_id="chat-b3", expert_id="max"),
            "in every chat with this Expert",
        ),
        (
            _session(session_id="chat-c", expert_id="frankie"),
            "for every Expert on your team",
        ),
    ]:
        assert _is_held(await _call(chat, _OPEN_WORLD, "do_thing", {"n": 3}))
        assert reason in gate.open_review.await_args.args[5]
    # The chat the rule was given in keeps its own word.
    assert not _is_held(await _call(max_, _OPEN_WORLD, "do_thing", {"n": 4}))
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
    subject = gate.open_review.await_args.args[6]
    assert subject.irreversible
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


async def _answer_with_rule(
    gate,
    rule: chat_rules.ChatRule,
    session: ChatSession | None = None,
    scope: chat_rules.Scope = "chat",
) -> None:
    """What the approve endpoint does for the card the gate just opened."""
    session = session or _session()
    subject = gate.open_review.await_args.args[6]
    row = SimpleNamespace(status=ReviewStatus.APPROVED)
    chat = SimpleNamespace(expert_id=session.expert_id)
    with patch(
        f"{_GATE}.chat_rules.get_chat_session_metadata", AsyncMock(return_value=chat)
    ):
        await chat_rules.set_answer_rules(
            session.session_id,
            "user-1",
            {"r": row},
            {"r": rule},
            {"r": subject.key},
            {"r": scope},
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


async def _opened(
    review_id, user_id, session, tool_name, args, reason, subject=None, **_
) -> Headline:
    # The headline the real card stores, so the chat row names what it names.
    return Headline.model_validate(review_payload(tool_name, args, subject)["headline"])
