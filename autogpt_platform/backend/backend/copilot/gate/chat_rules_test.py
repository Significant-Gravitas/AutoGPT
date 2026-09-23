from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.gate import chat_rules


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


def _row(status: ReviewStatus, key: str | None = "mcp:h/t") -> SimpleNamespace:
    payload = {"tool": "run_capability", "arguments": {}}
    if key:
        payload["subject"] = {"key": key, "name": "t on h", "effect": "external"}
    return SimpleNamespace(status=status, payload=payload)


async def test_an_approved_card_rules_on_the_subject_it_named(redis):
    await chat_rules.set_answer_rules(
        "s", {"a": _row(ReviewStatus.APPROVED)}, {"a": "allow"}
    )
    assert await chat_rules.rule_for("s", "mcp:h/t") == "allow"
    assert await chat_rules.rule_for("s", "mcp:h/other") is None


@pytest.mark.parametrize(
    "row",
    [
        _row(ReviewStatus.REJECTED),
        _row(ReviewStatus.WAITING),
        # A bare tool or a held read names no subject, so a click cannot rule on it.
        _row(ReviewStatus.APPROVED, key=None),
    ],
)
async def test_only_an_approved_subject_card_sets_a_rule(redis, row):
    await chat_rules.set_answer_rules("s", {"a": row}, {"a": "allow"})
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
        assert await chat_rules.rule_for("s", "mcp:h/t") == "ask"
