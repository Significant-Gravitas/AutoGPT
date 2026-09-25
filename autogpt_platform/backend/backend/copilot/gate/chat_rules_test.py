from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.gate import chat_rules, held


class _Redis:
    def __init__(self, data: dict[str, str] | None = None) -> None:
        self.data = data or {}

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.data[key] = value

    async def get(self, key: str) -> str | None:
        return self.data.get(key)


@pytest.fixture
def redis():
    store = _Redis()
    with patch.object(chat_rules, "get_redis_async", AsyncMock(return_value=store)):
        yield store


def _row(status: ReviewStatus) -> SimpleNamespace:
    return SimpleNamespace(status=status)


async def test_an_approved_card_rules_on_the_subject_it_named(redis):
    await chat_rules.set_answer_rules(
        "s", "u", {"a": _row(ReviewStatus.APPROVED)}, {"a": "allow"}, {"a": "mcp:h/t"}
    )
    assert await _rule("s", "mcp:h/t") == "allow"
    assert await _rule("s", "mcp:h/other") is None
    assert await _rule("other-chat", "mcp:h/t") is None


@pytest.mark.parametrize(
    "status, keys",
    [
        (ReviewStatus.REJECTED, {"a": "mcp:h/t"}),
        (ReviewStatus.WAITING, {"a": "mcp:h/t"}),
        # A bare tool or a held read names no subject, so a click cannot rule on it.
        (ReviewStatus.APPROVED, {}),
    ],
)
async def test_only_an_approved_subject_card_sets_a_rule(redis, status, keys):
    await chat_rules.set_answer_rules(
        "s", "u", {"a": _row(status)}, {"a": "allow"}, keys, {"a": "team"}
    )
    assert redis.data == {}


async def test_a_later_answer_replaces_the_rule(redis):
    await chat_rules.set_ask("s", "mcp:h/t", "u", None)
    await chat_rules.set_rule("s", "mcp:h/t", "judge")
    assert await _rule("s", "mcp:h/t") == "judge"


async def test_a_rule_written_before_decisions_existed_still_asks():
    store = _Redis({"copilot:gate:ask:s:bash_exec": "1"})
    with patch.object(chat_rules, "get_redis_async", AsyncMock(return_value=store)):
        assert await _rule("s", "bash_exec") == "ask"


async def test_an_unreadable_store_asks():
    with patch.object(
        chat_rules, "get_redis_async", AsyncMock(side_effect=ConnectionError)
    ):
        assert await _rule("s", "mcp:h/t") == "unreadable"


@pytest.mark.parametrize(
    "narrow, wide", [("chat", "expert"), ("chat", "team"), ("expert", "team")]
)
@pytest.mark.parametrize("narrow_rule, wide_rule", [("ask", "allow"), ("allow", "ask")])
async def test_the_narrowest_rule_decides(redis, narrow, wide, narrow_rule, wide_rule):
    await _set(wide, wide_rule)
    await _set(narrow, narrow_rule)
    assert await _rule("s", "mcp:h/t", "frankie") == narrow_rule


async def test_an_expert_rule_holds_only_in_that_experts_chats(redis):
    await _set("expert", "allow")
    assert await _rule("other-chat", "mcp:h/t", "frankie") == "allow"
    assert await _rule("other-chat", "mcp:h/t", "maria") is None
    assert await _rule("other-chat", "mcp:h/t", None) is None


async def test_a_rejection_revokes_every_wider_rule_that_would_have_run_it(redis):
    await _set("expert", "allow")
    await _set("team", "judge")
    await chat_rules.set_ask("s", "mcp:h/t", "u", "frankie")

    assert await _rule("s", "mcp:h/t", "frankie") == "ask"
    assert await _rule("other-chat", "mcp:h/t", "frankie") == "ask"
    assert await _rule("other-chat", "mcp:h/t", "maria") == "ask"


async def test_a_rejection_writes_no_wider_rule_where_there_was_none(redis):
    await chat_rules.set_ask("s", "mcp:h/t", "u", "frankie")
    assert await _rule("other-chat", "mcp:h/t", "frankie") is None


@pytest.mark.parametrize(
    "scope, otto, reason",
    [
        ("chat", False, chat_rules.DECLINED),
        ("expert", False, "You declined this in every chat with this Expert on 5 Sep."),
        ("expert", True, "You declined this in every chat with Otto on 5 Sep."),
        ("team", False, "You declined this for every Expert on your team on 5 Sep."),
    ],
)
def test_a_decline_names_its_scope_and_day(scope, otto, reason):
    since = None if scope == "chat" else date(2026, 9, 5)
    hit = chat_rules.RuleHit(rule="ask", scope=scope, since=since, otto=otto)
    assert hit.reason == reason


async def test_only_a_held_call_with_a_subject_offers_a_key():
    calls = {
        "mcp": held.HeldCall(
            review_id="mcp",
            tool_name="run_capability",
            tool_call_id="c",
            args={},
            rule_key="mcp:h/t",
        ),
        "tool": held.HeldCall(
            review_id="tool",
            tool_name="bash_exec",
            tool_call_id="c",
            args={},
            rule_key="bash_exec",
        ),
        "read": held.HeldCall(
            review_id="read", tool_name="web_fetch", tool_call_id="c", args={}
        ),
    }
    with patch.object(held, "_held", AsyncMock(return_value=calls)):
        keys = await held.subject_keys("s", ["mcp", "tool", "read", "gone"])
    assert keys == {"mcp": "mcp:h/t"}


async def test_an_unreadable_store_approves_without_a_rule():
    with patch.object(held, "_held", AsyncMock(side_effect=ConnectionError)):
        assert await held.subject_keys("s", ["mcp"]) == {}


async def _set(scope: str, rule: chat_rules.ChatRule) -> None:
    if scope == "chat":
        await chat_rules.set_rule("s", "mcp:h/t", rule)
    else:
        await chat_rules.set_scoped_rule(scope, "u", "frankie", "mcp:h/t", rule)


async def _rule(session_id: str, key: str, expert_id: str | None = None):
    hit = await chat_rules.rule_for(session_id, key, "u", expert_id)
    return hit.rule if hit else None
