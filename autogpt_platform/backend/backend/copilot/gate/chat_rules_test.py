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
    assert await chat_rules.rule_for("s", "mcp:h/t", "u") == ("allow", None)
    assert await chat_rules.rule_for("s", "mcp:h/other", "u") is None
    assert await chat_rules.rule_for("other-chat", "mcp:h/t", "u") is None


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
        "s", "u", {"a": _row(status)}, {"a": "allow"}, keys, frozenset({"a"})
    )
    assert redis.data == {}


async def test_a_later_answer_replaces_the_rule(redis):
    await chat_rules.set_ask("s", "mcp:h/t", "u")
    await chat_rules.set_rule("s", "mcp:h/t", "judge")
    assert await chat_rules.rule_for("s", "mcp:h/t", "u") == ("judge", None)


async def test_a_rule_written_before_decisions_existed_still_asks():
    store = _Redis({"copilot:gate:ask:s:bash_exec": "1"})
    with patch.object(chat_rules, "get_redis_async", AsyncMock(return_value=store)):
        assert await chat_rules.rule_for("s", "bash_exec", "u") == ("ask", None)


async def test_an_unreadable_store_asks():
    with patch.object(
        chat_rules, "get_redis_async", AsyncMock(side_effect=ConnectionError)
    ):
        assert await chat_rules.rule_for("s", "mcp:h/t", "u") == ("unreadable", None)


async def test_a_rejection_revokes_a_team_rule_and_says_so(redis):
    await chat_rules.set_team_rule("u", "mcp:h/t", "allow")
    await chat_rules.set_ask("chat-b", "mcp:h/t", "u")

    hit = await chat_rules.rule_for("chat-c", "mcp:h/t", "u")
    assert hit is not None and hit.rule == "ask"
    assert hit.team_since is not None
    assert hit.reason.startswith("You declined this for all your experts on ")


async def test_a_rejection_without_a_team_rule_stays_in_its_chat(redis):
    await chat_rules.set_ask("chat-b", "mcp:h/t", "u")
    assert await chat_rules.rule_for("chat-c", "mcp:h/t", "u") is None


def test_a_team_decline_names_the_day_it_was_given():
    hit = chat_rules.RuleHit("ask", date(2026, 9, 5))
    assert hit.reason == "You declined this for all your experts on 5 Sep."
    assert chat_rules.RuleHit("ask").reason == chat_rules.DECLINED


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
