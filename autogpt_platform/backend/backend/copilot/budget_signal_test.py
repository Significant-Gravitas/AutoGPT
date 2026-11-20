"""Tests for the budget signals a turn's model reads.

The ledger is exercised through the same in-memory Redis ``tree_test`` uses,
so the two suites cannot drift on hash semantics.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from typing import cast

import pytest

from backend.data.redis_client import AsyncRedisClient

from . import budget_signal
from .budget_signal import (
    CHECKPOINT_INSTRUCTION,
    build_spawn_state_note,
    build_turn_budget_block,
)
from .tree import TreeLedger, root_envelope
from .tree_test import BrokenRedis, FakeRedis

ENVELOPE = root_envelope("t")


@pytest.fixture
def ledger(monkeypatch) -> TreeLedger:
    """A ledger on fake Redis, wired into both module entry points."""
    live = TreeLedger(cast(AsyncRedisClient, FakeRedis()))

    async def _get() -> TreeLedger:
        return live

    monkeypatch.setattr(budget_signal, "get_tree_ledger", _get)

    async def _limits(_uid, _d, _w):
        return 1_000_000, 7_000_000, "BASIC"

    async def _remaining(**_kw):
        return 0.58

    monkeypatch.setattr(budget_signal, "get_global_rate_limits", _limits)
    monkeypatch.setattr(budget_signal, "get_remaining_usd_budget", _remaining)
    return live


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "envelope, flag",
    [(None, True), (ENVELOPE, False)],
    ids=["no-envelope", "flag-off"],
)
async def test_the_prompt_is_untouched_when_the_signal_does_not_apply(
    ledger, monkeypatch, envelope, flag
) -> None:
    """Empty output alone would not prove the guard: the module swallows every
    error, so a guard removed and a guard held both return ``""``. What
    separates them is whether the ledger was read at all."""
    monkeypatch.setattr(budget_signal.config, "tree_budget_signal_enabled", flag)
    reads: list[int] = []

    async def _counted() -> TreeLedger:
        reads.append(1)
        raise AssertionError("read the ledger for a turn that has no signal")

    monkeypatch.setattr(budget_signal, "get_tree_ledger", _counted)

    assert await build_turn_budget_block(envelope, "u") == ""
    assert not reads


@pytest.mark.asyncio
async def test_the_block_states_both_budgets_and_the_seats_left(ledger) -> None:
    await ledger.open("t", ceiling_microdollars=500_000, max_nodes=8, initial_nodes=3)
    await ledger.charge("t", 190_000)

    block = await build_turn_budget_block(ENVELOPE, "u")

    assert block.startswith("<budget_status>\n")
    assert block.endswith("</budget_status>\n\n")
    assert "$0.31 of its $0.50 budget left" in block
    assert "5 more sub-sessions" in block
    assert "$0.58 of today's budget left" in block
    assert CHECKPOINT_INSTRUCTION not in block


@pytest.mark.asyncio
async def test_a_tree_that_has_not_spawned_yet_shows_its_prospective_ceiling(
    ledger, monkeypatch
) -> None:
    async def _ceiling(_user_id):
        return 500_000

    monkeypatch.setattr(budget_signal, "resolve_root_ceiling_microdollars", _ceiling)
    monkeypatch.setattr(budget_signal.config, "tree_max_nodes", 8)

    block = await build_turn_budget_block(ENVELOPE, "u")

    assert "$0.50 of its $0.50 budget left" in block
    assert "7 more sub-sessions" in block


@pytest.mark.asyncio
async def test_a_tier_that_may_not_spend_is_not_offered_sub_sessions(ledger) -> None:
    """A zero ceiling is a tier that may not spend at all, so a line offering
    seats contradicts the refusal the spawn would hit in the same turn."""
    await ledger.open("t", ceiling_microdollars=0, max_nodes=8, initial_nodes=1)

    block = await build_turn_budget_block(ENVELOPE, "u")

    assert "no subscription" in block
    assert "more sub-sessions" not in block
    assert "$0.00 of its $0.00" not in block


@pytest.mark.asyncio
async def test_the_daily_line_is_dropped_when_there_is_no_user(ledger) -> None:
    await ledger.open("t", ceiling_microdollars=500_000, max_nodes=8, initial_nodes=1)
    assert "today's budget" not in await build_turn_budget_block(ENVELOPE, None)


@pytest.mark.asyncio
async def test_the_checkpoint_fires_once_per_tree_and_not_before_the_threshold(
    ledger,
) -> None:
    await ledger.open("t", ceiling_microdollars=500_000, max_nodes=8, initial_nodes=2)
    await ledger.charge("t", 390_000)
    assert CHECKPOINT_INSTRUCTION not in await build_turn_budget_block(ENVELOPE, "u")

    await ledger.charge("t", 10_000)
    assert CHECKPOINT_INSTRUCTION in await build_turn_budget_block(ENVELOPE, "u")
    assert CHECKPOINT_INSTRUCTION not in await build_turn_budget_block(ENVELOPE, "u")


@pytest.mark.asyncio
async def test_two_turns_crossing_at_once_still_get_one_checkpoint(ledger) -> None:
    await ledger.open("t", ceiling_microdollars=500_000, max_nodes=8, initial_nodes=2)
    await ledger.charge("t", 400_000)

    blocks = await asyncio.gather(
        build_turn_budget_block(ENVELOPE, "u"),
        build_turn_budget_block(ENVELOPE, "u"),
    )

    assert sum(CHECKPOINT_INSTRUCTION in b for b in blocks) == 1


@pytest.mark.asyncio
async def test_a_brownout_drops_the_daily_line_and_a_real_zero_still_states_it(
    ledger, monkeypatch
) -> None:
    """``get_remaining_usd_budget`` answers a brown-out with the floor it was
    handed, so a caller asking for ``0.0`` cannot tell "unknown" from "spent to
    the last cent" — and the second is the one worth telling the model."""
    await ledger.open("t", ceiling_microdollars=500_000, max_nodes=8, initial_nodes=1)

    async def _brownout(*, floor_usd: float, **_kw) -> float:
        return floor_usd

    monkeypatch.setattr(budget_signal, "get_remaining_usd_budget", _brownout)
    assert "today's budget" not in await build_turn_budget_block(ENVELOPE, "u")

    async def _spent_out(**_kw) -> float:
        return 0.0

    monkeypatch.setattr(budget_signal, "get_remaining_usd_budget", _spent_out)
    assert "$0.00 of today's budget left" in await build_turn_budget_block(
        ENVELOPE, "u"
    )


@pytest.mark.asyncio
async def test_a_dead_ledger_never_fails_the_turn(monkeypatch) -> None:
    async def _get() -> TreeLedger:
        return TreeLedger(cast(AsyncRedisClient, BrokenRedis()))

    monkeypatch.setattr(budget_signal, "get_tree_ledger", _get)
    assert await build_turn_budget_block(ENVELOPE, "u") == ""
    monkeypatch.setattr(budget_signal, "get_current_envelope", lambda: ENVELOPE)
    assert await build_spawn_state_note() == ""


@pytest.mark.asyncio
async def test_the_spawn_note_carries_what_was_spent_and_what_remains(
    ledger, monkeypatch
) -> None:
    monkeypatch.setattr(budget_signal, "get_current_envelope", lambda: ENVELOPE)
    await ledger.open("t", ceiling_microdollars=500_000, max_nodes=8, initial_nodes=1)
    await ledger.charge("t", 420_000)
    await ledger.admit(ENVELOPE)

    note = await build_spawn_state_note()

    assert "$0.42 spent of $0.50" in note
    assert "$0.08 left" in note
    assert "2 of 8 sub-sessions used (6 left)" in note


@pytest.mark.asyncio
async def test_the_spawn_note_is_empty_without_a_tree(ledger, monkeypatch) -> None:
    monkeypatch.setattr(budget_signal, "get_current_envelope", lambda: None)
    assert await build_spawn_state_note() == ""
    # An envelope whose tree never opened a ledger has nothing to report.
    monkeypatch.setattr(budget_signal, "get_current_envelope", lambda: ENVELOPE)
    assert await build_spawn_state_note() == ""


# ── engine wiring ──────────────────────────────────────────────────────
# Both stream functions need a live database, a model route and an LLM to
# reach their injection point, so the seam is pinned by inspection — the
# pattern ``markers_test`` already uses on the same two functions.


def _stream_sources() -> dict[str, str]:
    from backend.copilot.baseline.service import stream_chat_completion_baseline
    from backend.copilot.sdk.service import stream_chat_completion_sdk

    return {
        "baseline": inspect.getsource(stream_chat_completion_baseline),
        "sdk": inspect.getsource(stream_chat_completion_sdk),
    }


@pytest.mark.parametrize("engine", ["baseline", "sdk"])
def test_both_engines_build_the_block_every_turn(engine: str) -> None:
    assert "build_turn_budget_block(" in _stream_sources()[engine]


@pytest.mark.parametrize(
    ("engine", "transcribed"),
    [("baseline", "user_message_for_transcript"), ("sdk", "current_message")],
)
def test_the_block_never_reaches_the_variable_the_transcript_records(
    engine: str, transcribed: str
) -> None:
    """The transcript is replayed on the next turn; one stale figure per turn
    is what folding the block into it would leave behind. Both engines reach
    the block through a ``budget_status`` local, so naming only the builder
    would miss every regression that actually folds it in."""
    tree = ast.parse(textwrap.dedent(_stream_sources()[engine]))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if transcribed not in targets:
            continue
        folded = ast.unparse(node.value)
        for carrier in ("build_turn_budget_block", "budget_status"):
            assert (
                carrier not in folded
            ), f"{engine}: {carrier} was assigned into {transcribed}"
