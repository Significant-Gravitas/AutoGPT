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
        "s", {"a": _row(ReviewStatus.APPROVED)}, {"a": "allow"}, {"a": "mcp:h/t"}
    )
    assert await chat_rules.rule_for("s", "mcp:h/t") == "allow"
    assert await chat_rules.rule_for("s", "mcp:h/other") is None


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
    await chat_rules.set_answer_rules("s", {"a": _row(status)}, {"a": "allow"}, keys)
    assert redis.data == {}


async def test_a_later_answer_replaces_the_rule(redis):
    await chat_rules.set_ask("s", "mcp:h/t")
    await chat_rules.set_rule("s", "mcp:h/t", "judge")
    assert await chat_rules.rule_for("s", "mcp:h/t") == "judge"


async def test_a_rule_written_before_decisions_existed_still_asks():
    store = _Redis({"copilot:gate:ask:s:bash_exec": "1"})
    with patch.object(chat_rules, "get_redis_async", AsyncMock(return_value=store)):
        assert await chat_rules.rule_for("s", "bash_exec") == "ask"


async def test_an_unreadable_store_asks():
    with patch.object(
        chat_rules, "get_redis_async", AsyncMock(side_effect=ConnectionError)
    ):
        assert await chat_rules.rule_for("s", "mcp:h/t") == "unreadable"


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
