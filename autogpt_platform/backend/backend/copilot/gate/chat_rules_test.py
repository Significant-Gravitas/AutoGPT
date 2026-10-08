import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.gate import chat_rules, held


class _Redis:
    def __init__(self, data: dict[str, str] | None = None) -> None:
        self.data = data or {}
        self.ttls: dict[str, int | None] = {}

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.data[key] = value
        self.ttls[key] = ttl

    async def set(self, key: str, value: str) -> None:
        self.data[key] = value
        self.ttls[key] = None

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def delete(self, *keys: str) -> int:
        return sum(1 for key in keys if self.data.pop(key, None) is not None)


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
        # A held read or a money card offers no key, so a click cannot rule on it.
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


async def test_a_held_call_offers_the_key_its_card_can_rule_on():
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
        # Parked over the spend ceiling: reads never consult a rule.
        "paid": held.HeldCall(
            review_id="paid",
            tool_name="consult_teammate",
            tool_call_id="c",
            args={},
            rule_key="consult_teammate",
        ),
    }
    with patch.object(held, "_held", AsyncMock(return_value=calls)):
        keys = await held.subject_keys("s", ["mcp", "tool", "read", "paid", "gone"])
    assert keys == {"mcp": "mcp:h/t", "tool": "bash_exec"}


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


# --- Allow rules with a lifetime ("Always allow" and its variants) ---


async def _grant(
    rule: chat_rules.ChatRule = "allow",
    session_id: str = "s",
    expert_id: str | None = "frankie",
    **grant,
) -> None:
    await chat_rules.set_granted_rule(
        session_id, "u", expert_id, "mcp:h/t", rule, chat_rules.AllowGrant(**grant)
    )


async def _hit(session_id: str = "s", **kwargs) -> chat_rules.RuleHit | None:
    return await chat_rules.rule_for(session_id, "mcp:h/t", "u", "frankie", **kwargs)


async def test_always_allow_defaults_to_this_chat_for_good(redis):
    await _grant()
    hit = await _hit()
    assert hit is not None and hit.rule == "allow"
    assert (hit.scope, hit.lifetime) == ("chat", "always")
    assert await _hit("other-chat") is None


async def test_once_is_spent_by_the_first_call_it_covers(redis):
    await _grant(lifetime="once")
    first = await _hit()
    assert first is not None and first.rule == "allow"
    assert await _hit() is None


async def test_a_spent_once_falls_through_to_a_wider_rule(redis):
    await chat_rules.set_scoped_rule("team", "u", None, "mcp:h/t", "judge")
    await _grant(lifetime="once")
    assert (await _hit()).rule == "allow"
    hit = await _hit()
    assert hit is not None and (hit.rule, hit.scope) == ("judge", "team")


async def test_a_turn_rule_holds_for_the_task_that_first_uses_it(redis):
    await _grant(lifetime="turn")
    assert (await _hit(turn_id="task-1")).rule == "allow"
    assert (await _hit(turn_id="task-1")).rule == "allow"
    assert await _hit(turn_id="task-2") is None
    # Gone once another task met it, so the first task cannot get it back.
    assert await _hit(turn_id="task-1") is None


async def test_a_turn_rule_with_no_task_to_bind_is_spent_like_once(redis):
    await _grant(lifetime="turn")
    assert (await _hit()).rule == "allow"
    assert await _hit() is None


async def test_a_ttl_rule_expires_after_its_hours(redis):
    await _grant(scope="team", lifetime="ttl", ttl_hours=2)
    key = chat_rules._scoped_key("team", "u", None, "mcp:h/t")
    assert redis.ttls[key] == 2 * 60 * 60
    hit = await _hit("other-chat")
    assert hit is not None and (hit.rule, hit.scope, hit.lifetime) == (
        "allow",
        "team",
        "ttl",
    )
    stored = json.loads(redis.data[key])
    stored["expires_at"] = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    redis.data[key] = json.dumps(stored)
    assert await _hit("other-chat") is None
    assert key not in redis.data


async def test_always_at_a_wider_scope_never_ages_out(redis):
    await _grant(scope="expert", lifetime="always")
    key = chat_rules._scoped_key("expert", "u", "frankie", "mcp:h/t")
    assert redis.ttls[key] is None
    hit = await _hit("other-chat")
    assert hit is not None and (hit.scope, hit.lifetime) == ("expert", "always")


@pytest.mark.parametrize("lifetime", ["once", "turn", "chat"])
async def test_a_chat_bound_lifetime_never_widens_past_its_chat(redis, lifetime):
    await _grant(scope="team", lifetime=lifetime)
    assert not any(":team:" in key for key in redis.data)
    assert await _hit("other-chat", turn_id="t") is None


async def test_a_decline_replaces_an_always_allow_at_every_scope(redis):
    await _grant(scope="expert", lifetime="always")
    await _grant()
    await chat_rules.set_ask("s", "mcp:h/t", "u", "frankie")
    assert (await _hit()).rule == "ask"
    assert (await _hit("other-chat")).rule == "ask"


async def test_a_later_always_allow_in_this_chat_replaces_its_decline(redis):
    await chat_rules.set_ask("s", "mcp:h/t", "u", "frankie")
    await _grant()
    assert (await _hit()).rule == "allow"


async def test_a_delegated_chat_inherits_declines_but_not_allows(redis):
    await chat_rules.set_ask("parent", "mcp:h/t", "u", None)
    await _grant(session_id="child")
    hit = await _hit("child", ancestors=["parent"])
    assert hit is not None and hit.rule == "ask" and hit.inherited
    assert hit.reason == chat_rules.INHERITED

    await _grant(session_id="parent-2")
    assert await _hit("child-2", ancestors=["parent-2"]) is None


async def test_a_delegated_chat_may_be_stricter_than_its_parent(redis):
    await _grant(session_id="parent")
    await chat_rules.set_ask("child", "mcp:h/t", "u", None)
    assert (await _hit("child", ancestors=["parent"])).rule == "ask"


async def test_the_chain_of_parents_is_walked_and_bounded():
    chain = {
        "c1": "c2",
        "c2": "c3",
        "c3": "c4",
        "c4": "c5",
    }

    async def meta(session_id: str, user_id: str):
        return SimpleNamespace(
            metadata=SimpleNamespace(delegated_by_session_id=chain.get(session_id))
        )

    with patch.object(chat_rules, "get_chat_session_metadata", meta):
        assert await chat_rules.ancestors_of("c0", "c1", "u") == ["c1", "c2", "c3"]


async def test_an_answer_with_a_grant_saves_its_lifetime(redis):
    with patch.object(
        chat_rules,
        "get_chat_session_metadata",
        AsyncMock(return_value=SimpleNamespace(expert_id="frankie")),
    ):
        await chat_rules.set_answer_rules(
            "s",
            "u",
            {"a": _row(ReviewStatus.APPROVED), "b": _row(ReviewStatus.APPROVED)},
            {"a": "allow", "b": "allow"},
            {"a": "mcp:h/t", "b": "mcp:h/other"},
            {"a": "expert", "b": "chat"},
            {"a": chat_rules.AllowGrant(scope="expert", lifetime="ttl", ttl_hours=3)},
        )
    hit = await _hit("other-chat")
    assert hit is not None and (hit.scope, hit.lifetime) == ("expert", "ttl")
    # Without a grant the rule is the legacy chat-long one.
    legacy = await chat_rules.rule_for("s", "mcp:h/other", "u", "frankie")
    assert legacy is not None and (legacy.rule, legacy.lifetime) == ("allow", None)


def test_an_unreadable_stored_rule_asks():
    stored = chat_rules._parse("{not json", legacy_scoped=False)
    assert stored.rule == "ask"
