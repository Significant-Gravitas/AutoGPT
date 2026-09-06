"""What a turn's model is told about the budget it is spending.

Three signals, one wording, so both engines and all three spawn tools agree.
Every number comes from the tree ledger (:mod:`backend.copilot.tree`) and the
per-user rate limit; this module is the phrasing and the thresholds.

- ``<budget_status>`` — one line prepended to every turn's message: what the
  tree has left, how many sub-sessions it may still start, what the account
  has left today. Live, so it is right from turn two onward.
- The wrap-up checkpoint — appended to that block once per tree, the first
  turn the tree crosses ``tree_wrapup_threshold`` of its ceiling.
- The spawn-result note — the same tree state on every spawn tool result, so
  the parent decides its next spawn from numbers rather than a guess.

Nothing here may fail a turn: every entry point returns ``""`` on any error,
and returns ``""`` unconditionally when the feature is off or the turn has no
tree envelope, which keeps the prompt byte-identical to what it was before.
"""

import logging

from backend.copilot.config import ChatConfig
from backend.copilot.context import get_current_envelope
from backend.copilot.rate_limit import get_global_rate_limits, get_remaining_usd_budget
from backend.copilot.tree import (
    TurnEnvelope,
    get_tree_ledger,
    resolve_root_ceiling_microdollars,
)

logger = logging.getLogger(__name__)
config = ChatConfig()

CHECKPOINT_INSTRUCTION = (
    "Most of this task's budget is spent. Checkpoint now: finish the current "
    "step, then list up to 3 short bullets of the most impactful remaining "
    "work. Start no new sub-sessions."
)


async def build_turn_budget_block(
    envelope: TurnEnvelope | None, user_id: str | None
) -> str:
    """The ``<budget_status>`` block for this turn, or ``""``.

    Prepended to the turn's message rather than the system prompt: the system
    prompt must stay identical across users for the cross-session prefix cache
    (see ``get_sdk_supplement``). Not persisted either — a budget line is true
    of one turn, and a replayed transcript would carry a stale one.
    """
    if not config.tree_budget_signal_enabled or envelope is None:
        return ""
    try:
        ceiling, spent, nodes, max_nodes = await _tree_state(envelope, user_id)
        line = (
            f"This task has ${_usd(ceiling - spent)} of its ${_usd(ceiling)} "
            f"budget left and can start {max(0, max_nodes - nodes)} more "
            f"sub-sessions."
        )
        daily = await _remaining_daily_usd(user_id)
        if daily is not None:
            line += f" This account has ${daily:.2f} of today's budget left."
        if await _crossed_wrapup(envelope, ceiling, spent):
            line += f"\n{CHECKPOINT_INSTRUCTION}"
        return f"<budget_status>\n{line}\n</budget_status>\n\n"
    except Exception:
        logger.warning("[budget] turn block unavailable", exc_info=True)
        return ""


async def build_spawn_state_note() -> str:
    """The tree's state after a spawn, for the spawn tool's own result.

    Reads the envelope off the contextvar rather than a parameter: the running
    turn is the spawner, so its ``tree_id`` is the tree the child just charged.
    """
    envelope = get_current_envelope()
    if not config.tree_budget_signal_enabled or envelope is None:
        return ""
    try:
        ledger = await get_tree_ledger()
        snapshot = await ledger.snapshot(envelope.tree_id)
        if not snapshot:
            return ""
        ceiling = snapshot.get("ceiling", 0)
        spent = snapshot.get("spent", 0)
        nodes = snapshot.get("nodes", 0)
        max_nodes = snapshot.get("max_nodes", 0)
        return (
            f" Task budget: ${_usd(spent)} spent of ${_usd(ceiling)} "
            f"(${_usd(ceiling - spent)} left); {nodes} of {max_nodes} "
            f"sub-sessions used ({max(0, max_nodes - nodes)} left)."
        )
    except Exception:
        logger.warning("[budget] spawn state unavailable", exc_info=True)
        return ""


async def _tree_state(
    envelope: TurnEnvelope, user_id: str | None
) -> tuple[int, int, int, int]:
    """``(ceiling, spent, nodes, max_nodes)`` in microdollars and counts.

    A tree that has not spawned yet has no ledger, so its ceiling is resolved
    the same way ``open()`` will resolve it — the model gets the real number
    on the turn it is deciding whether to spawn at all, not the turn after.
    """
    ledger = await get_tree_ledger()
    snapshot = await ledger.snapshot(envelope.tree_id)
    if snapshot:
        return (
            snapshot.get("ceiling", 0),
            snapshot.get("spent", 0),
            snapshot.get("nodes", 0),
            snapshot.get("max_nodes", 0),
        )
    return (
        await resolve_root_ceiling_microdollars(user_id),
        0,
        1,
        config.tree_max_nodes,
    )


async def _remaining_daily_usd(user_id: str | None) -> float | None:
    """``None`` rather than ``$0.00`` when the figure is unknowable — an
    unlimited account, or a Redis brown-out that already fails the turn's
    pre-gate closed."""
    if not user_id:
        return None
    daily, weekly, _tier = await get_global_rate_limits(
        user_id,
        config.daily_cost_limit_microdollars,
        config.weekly_cost_limit_microdollars,
    )
    remaining = await get_remaining_usd_budget(
        user_id=user_id,
        daily_cost_limit=daily,
        weekly_cost_limit=weekly,
        floor_usd=0.0,
    )
    return None if remaining == float("inf") else remaining


async def _crossed_wrapup(envelope: TurnEnvelope, ceiling: int, spent: int) -> bool:
    if ceiling <= 0 or spent < config.tree_wrapup_threshold * ceiling:
        return False
    return await (await get_tree_ledger()).claim_wrapup(envelope.tree_id)


def _usd(microdollars: int) -> str:
    return f"{max(0, microdollars) / 1_000_000:.2f}"
