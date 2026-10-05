"""The ``SessionStart`` hook hands the model back what compaction dropped.

Compaction replaces the session's first user message, and the first-turn
blocks at its start, with a summary. With matcher ``compact`` the CLI runs
these callbacks right after it, one after another, and puts every one's
``additionalContext`` after the summary, joined in registration order. Each
callback's context is capped on its own, so the re-send goes out in parts,
one per callback.
"""

import pytest

from .after_compaction import AFTER_COMPACTION_MAX_PARTS
from .security_hooks import create_security_hooks


def _session_start_matcher(context_after_compaction):
    hooks = create_security_hooks(
        user_id="u1", context_after_compaction=context_after_compaction
    )
    (matcher,) = hooks["SessionStart"]
    return matcher


async def _fire(context_after_compaction, source: str) -> list[dict]:
    """Run every callback the way the CLI does: in order, awaiting each."""
    return [
        await hook(
            {"hook_event_name": "SessionStart", "source": source},
            None,
            {"signal": None},
        )
        for hook in _session_start_matcher(context_after_compaction).hooks
    ]


def _resent(part: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": part,
        }
    }


def test_not_registered_without_a_context_source():
    hooks = create_security_hooks(user_id="u1")

    assert "SessionStart" not in hooks


def test_one_bounded_callback_per_part_for_compaction():
    async def _context() -> list[str]:
        return ["ctx"]

    matcher = _session_start_matcher(_context)

    assert matcher.matcher == "compact"
    assert len(matcher.hooks) == AFTER_COMPACTION_MAX_PARTS
    # The CLI waits on these before it carries on after the summary.
    assert matcher.timeout is not None and matcher.timeout > 0


@pytest.mark.asyncio
async def test_compaction_hands_back_the_context():
    async def _context() -> list[str]:
        return ["<available_skills>\n- name: weekly-report\n</available_skills>"]

    first, *rest = await _fire(_context, "compact")

    assert first == _resent(
        "<available_skills>\n- name: weekly-report\n</available_skills>"
    )
    assert rest == [{}] * (AFTER_COMPACTION_MAX_PARTS - 1)


@pytest.mark.asyncio
async def test_parts_go_out_in_order_from_one_rebuild():
    rebuilds: list[int] = []

    async def _context() -> list[str]:
        rebuilds.append(1)
        return ["lead and skills", "team", "user"]

    results = await _fire(_context, "compact")

    assert results[:3] == [_resent("lead and skills"), _resent("team"), _resent("user")]
    assert results[3:] == [{}] * (AFTER_COMPACTION_MAX_PARTS - 3)
    assert rebuilds == [1]


@pytest.mark.asyncio
async def test_every_compaction_rebuilds():
    versions = iter([["first"], ["second"]])

    async def _context() -> list[str]:
        return next(versions)

    matcher = _session_start_matcher(_context)
    payload = {"hook_event_name": "SessionStart", "source": "compact"}
    sent = []
    for _ in range(2):
        for hook in matcher.hooks:
            result = await hook(payload, None, {"signal": None})
            if result:
                sent.append(result["hookSpecificOutput"]["additionalContext"])

    assert sent == ["first", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["startup", "resume", "clear"])
async def test_other_session_starts_send_nothing(source):
    calls: list[str] = []

    async def _context() -> list[str]:
        calls.append(source)
        return ["ctx"]

    assert await _fire(_context, source) == [{}] * AFTER_COMPACTION_MAX_PARTS
    assert calls == []


@pytest.mark.asyncio
async def test_nothing_to_resend_sends_nothing():
    async def _context() -> list[str]:
        return []

    assert await _fire(_context, "compact") == [{}] * AFTER_COMPACTION_MAX_PARTS


@pytest.mark.asyncio
async def test_a_failed_rebuild_does_not_fail_the_hooks():
    async def _context() -> list[str]:
        raise RuntimeError("understanding lookup failed")

    assert await _fire(_context, "compact") == [{}] * AFTER_COMPACTION_MAX_PARTS
