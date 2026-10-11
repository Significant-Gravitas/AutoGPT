"""The ``SessionStart`` hook hands the model back what compaction dropped.

Compaction replaces the session's first user message, and the first-turn
blocks at its start, with a summary. With matcher ``compact`` the CLI starts
these callbacks right after it, all at the same time, and puts every one's
``additionalContext`` after the summary in the order the callbacks answer.
Each callback's context is capped on its own, so the re-send goes out in
parts, one per callback.
"""

import asyncio

import pytest

from .after_compaction import AFTER_COMPACTION_MAX_PARTS
from .security_hooks import create_security_hooks


def _session_start_matcher(context_after_compaction):
    hooks = create_security_hooks(
        user_id="u1",
        context_after_compaction=context_after_compaction,
        allowed_tools=(),
    )
    (matcher,) = hooks["SessionStart"]
    return matcher


async def _fire_hooks(hooks, source: str = "compact", *, start_order=None):
    """Run one compaction's callbacks the way the CLI and SDK do: every
    callback in its own task, all in flight together.

    Returns each callback's result by part, and the parts in the order their
    callbacks answered. ``start_order`` is the order the callbacks start in
    (registration order when not given).
    """
    answered: list[int] = []

    async def run(part: int) -> dict:
        result = await hooks[part](
            {"hook_event_name": "SessionStart", "source": source},
            None,
            {"signal": None},
        )
        answered.append(part)
        return result

    order = list(start_order or range(len(hooks)))
    results = await asyncio.gather(*(run(part) for part in order))
    return [results[order.index(part)] for part in range(len(hooks))], answered


async def _fire(context_after_compaction, source: str) -> list[dict]:
    results, _ = await _fire_hooks(
        _session_start_matcher(context_after_compaction).hooks, source
    )
    return results


def _resent(part: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": part,
        }
    }


def test_not_registered_without_a_context_source():
    hooks = create_security_hooks(user_id="u1", allowed_tools=())

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

    hooks = _session_start_matcher(_context).hooks
    sent = []
    for _ in range(2):
        results, _ = await _fire_hooks(hooks)
        sent.append(
            [r["hookSpecificOutput"]["additionalContext"] for r in results if r]
        )

    assert sent == [["first"], ["second"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "start_order",
    [[0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]],
    ids=["registration", "reversed", "shuffled"],
)
async def test_parts_answer_in_part_order_whichever_callback_starts_first(start_order):
    rebuilds: list[int] = []

    async def _context() -> list[str]:
        rebuilds.append(1)
        await asyncio.sleep(0)
        return ["one", "two", "three", "four"]

    hooks = _session_start_matcher(_context).hooks

    results, answered = await _fire_hooks(hooks, start_order=start_order)

    assert results == [_resent(p) for p in ("one", "two", "three", "four")]
    assert answered == [0, 1, 2, 3]
    assert rebuilds == [1]


@pytest.mark.asyncio
async def test_a_second_compaction_started_in_reverse_gets_its_own_rebuild():
    versions = iter([["a1", "a2", "a3", "a4"], ["b1", "b2", "b3", "b4"]])

    async def _context() -> list[str]:
        await asyncio.sleep(0)
        return next(versions)

    hooks = _session_start_matcher(_context).hooks

    first, first_answered = await _fire_hooks(hooks)
    second, second_answered = await _fire_hooks(hooks, start_order=[3, 2, 1, 0])

    assert first == [_resent(p) for p in ("a1", "a2", "a3", "a4")]
    assert second == [_resent(p) for p in ("b1", "b2", "b3", "b4")]
    assert first_answered == second_answered == [0, 1, 2, 3]


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
