"""The ``SessionStart`` hook hands the model back what compaction dropped.

Compaction replaces the session's first user message, and the first-turn
blocks at its start, with a summary. With matcher ``compact`` the CLI runs this
hook right after it and puts its ``additionalContext`` after the summary.
"""

import pytest

from .security_hooks import create_security_hooks


def _session_start_matcher(context_after_compaction):
    hooks = create_security_hooks(
        user_id="u1", context_after_compaction=context_after_compaction
    )
    (matcher,) = hooks["SessionStart"]
    return matcher


async def _fire(context_after_compaction, source: str) -> dict:
    (hook,) = _session_start_matcher(context_after_compaction).hooks
    return await hook(
        {"hook_event_name": "SessionStart", "source": source},
        None,
        {"signal": None},
    )


def test_not_registered_without_a_context_source():
    hooks = create_security_hooks(user_id="u1")

    assert "SessionStart" not in hooks


def test_registered_for_compaction_with_a_bound():
    async def _context() -> str:
        return "ctx"

    matcher = _session_start_matcher(_context)

    assert matcher.matcher == "compact"
    # The CLI waits on this hook before it carries on after the summary.
    assert matcher.timeout is not None and matcher.timeout > 0


@pytest.mark.asyncio
async def test_compaction_hands_back_the_context():
    async def _context() -> str:
        return "<available_skills>\n- name: weekly-report\n</available_skills>"

    result = await _fire(_context, "compact")

    assert result == {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": (
                "<available_skills>\n- name: weekly-report\n</available_skills>"
            ),
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["startup", "resume", "clear"])
async def test_other_session_starts_send_nothing(source):
    calls: list[str] = []

    async def _context() -> str:
        calls.append(source)
        return "ctx"

    assert await _fire(_context, source) == {}
    assert calls == []


@pytest.mark.asyncio
async def test_nothing_to_resend_sends_nothing():
    async def _context() -> str:
        return ""

    assert await _fire(_context, "compact") == {}


@pytest.mark.asyncio
async def test_a_failed_rebuild_does_not_fail_the_hook():
    async def _context() -> str:
        raise RuntimeError("understanding lookup failed")

    assert await _fire(_context, "compact") == {}
