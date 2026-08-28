"""Per-tree bounds for spawned copilot turns.

Every copilot turn belongs to a *tree* rooted in a turn nobody spawned — a
human typing, a schedule firing, an ``AutoPilotBlock``. A turn that a tool
spawns (``run_sub_session``, ``delegate_to_expert``, ``handoff_to_expert``)
is a child in its spawner's tree. Two objects carry the bounds:

- :class:`TurnEnvelope` — per turn, immutable, travels on the queue entry.
  Derived by :func:`derive_child_envelope` from the *spawning turn's*
  envelope by operations that can only narrow (depth + 1 toward a cap, tool
  set intersected, deadline min, taint or-ed). Nothing a model supplies can
  widen it, and nothing reads it off a session row.
- :class:`TreeLedger` — per tree, mutable, one Redis hash. A turn is admitted
  at ``dispatch_turn`` only while the tree's metered spend is under its
  ceiling and its node count under its cap. Cost is charged after each turn
  by the same path that feeds the per-user rate limit.

The ledger meters; it does not reserve. That is deliberate: the SDK's
``max_budget_usd`` is a per-query stop that floors upward and does not exist
on the Codex transport, so a reservation could never be enforced — a counter
checked at the one seam every turn passes through can.
"""

import logging
from collections.abc import Awaitable
from datetime import UTC, datetime, timedelta
from typing import cast

from pydantic import BaseModel, Field

from backend.copilot.active_turns import MAX_TURN_LIFETIME_SECONDS
from backend.copilot.config import ChatConfig
from backend.copilot.permissions import ALL_TOOL_NAMES, CopilotPermissions
from backend.copilot.rate_limit import get_global_rate_limits, get_remaining_usd_budget
from backend.data.redis_client import AsyncRedisClient, get_redis_async

logger = logging.getLogger(__name__)
config = ChatConfig()

# One bound for every spawn kind; ``expert_delegation`` re-exports it as the
# chain bound. The provenance walk there only counts cross-expert hops — this
# counts isolates too.
MAX_DEPTH = 3

SPAWN_TOOLS: frozenset[str] = frozenset(
    {"run_sub_session", "delegate_to_expert", "handoff_to_expert"}
)

# Withheld from children unless the spawner grants them explicitly. Each one
# either outlives the tree, leaves the platform, binds a credential, or cannot
# be undone — none of which a spawned turn should get by default.
DESCENT_DENIED_TOOLS: frozenset[str] = frozenset(
    {
        "schedule_followup",
        "setup_agent_webhook_trigger",
        "update_preset",
        "store_skill",
        "memory_store",
        "add_understanding",
        "post_to_chat_platform",
        "connect_integration",
        "run_mcp_tool",
        "delete_folder",
        "delete_preset",
        "delete_schedule",
        "delete_skill",
        "delete_workspace_file",
        "hire_expert",
        "raise_expert",
        "update_expert",
        "confirm_expert_change",
    }
)

_LEDGER_KEY_PREFIX = "copilot:tree:"


class TreeRefusal(Exception):
    """A turn may not start; ``message`` is written for the model."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class TurnEnvelope(BaseModel):
    tree_id: str
    depth: int = 0
    tainted: bool = False
    # None = unrestricted root. A child always carries a concrete set.
    tools: frozenset[str] | None = None
    deadline_at: datetime | None = None

    def permits(self, tool_name: str) -> bool:
        return self.tools is None or tool_name in self.tools

    def as_permissions(self) -> CopilotPermissions | None:
        """The tool set as a whitelist, so the engines hide what this turn
        may not call. Hiding is presentation; ``BaseTool.execute`` enforces."""
        if self.tools is None:
            return None
        return CopilotPermissions(tools=sorted(self.tools), tools_exclude=False)


class SpawnRequest(BaseModel):
    """What a spawner asks for its child. Every field is clamped, never raised."""

    tools: list[str] | None = None
    may_spawn: bool = False
    max_seconds: int | None = Field(default=None, ge=1)
    born_tainted: bool = False


def root_envelope(turn_id: str, *, tainted: bool = False) -> TurnEnvelope:
    return TurnEnvelope(tree_id=turn_id, depth=0, tainted=tainted)


def derive_child_envelope(
    spawner: TurnEnvelope,
    request: SpawnRequest,
    *,
    spawner_permissions: CopilotPermissions | None = None,
    now: datetime | None = None,
) -> TurnEnvelope:
    """The child's envelope, or :class:`TreeRefusal`. Pure.

    ``spawner_permissions`` is the spawning turn's own capability filter
    (an ``AutoPilotBlock`` whitelist, say). It lives beside the envelope
    rather than in it today, so it is folded in here to keep the ceiling
    honest.
    """
    now = now or datetime.now(UTC)
    depth = spawner.depth + 1
    if depth > MAX_DEPTH:
        raise TreeRefusal(
            f"This task is already {spawner.depth} hops deep; do as much as you "
            "can yourself instead of passing it on again."
        )
    if spawner.deadline_at is not None and spawner.deadline_at <= now:
        raise TreeRefusal("This turn's deadline has passed; report what you have.")
    if not any(spawner.permits(t) for t in SPAWN_TOOLS):
        raise TreeRefusal("This turn may not spawn further work.")

    ceiling = spawner.tools if spawner.tools is not None else ALL_TOOL_NAMES
    if spawner_permissions is not None:
        ceiling = ceiling & spawner_permissions.effective_allowed_tools(ALL_TOOL_NAMES)
    requested = (
        frozenset(request.tools)
        if request.tools is not None
        else ALL_TOOL_NAMES - DESCENT_DENIED_TOOLS
    )
    tools = ceiling & requested
    if not request.may_spawn:
        tools = tools - SPAWN_TOOLS

    deadline_at = spawner.deadline_at
    if request.max_seconds is not None:
        candidate = now + timedelta(seconds=request.max_seconds)
        deadline_at = candidate if deadline_at is None else min(deadline_at, candidate)

    return TurnEnvelope(
        tree_id=spawner.tree_id,
        depth=depth,
        tainted=spawner.tainted or request.born_tainted,
        tools=tools,
        deadline_at=deadline_at,
    )


class TreeLedger:
    """One Redis hash per tree: ``ceiling``, ``spent``, ``nodes``, ``max_nodes``.

    Admission is increment-then-check with rollback, the same non-locked
    pattern ``acquire_turn_slot`` uses: two racing spawns can both pass by
    one, and that over-admit is the bound's stated slack. Spend is compared
    against a counter charged after each turn, so the overshoot per tree is
    one turn's cost — the SDK per-query cap on the platform transport.
    """

    def __init__(self, redis: AsyncRedisClient) -> None:
        self._redis = redis

    @staticmethod
    def key(tree_id: str) -> str:
        return f"{_LEDGER_KEY_PREFIX}{tree_id}"

    async def open(
        self, tree_id: str, *, ceiling_microdollars: int, max_nodes: int
    ) -> None:
        key = self.key(tree_id)
        for field, value in (
            ("ceiling", max(0, ceiling_microdollars)),
            ("spent", 0),
            ("nodes", 0),
            ("max_nodes", max(1, max_nodes)),
        ):
            await self._hsetnx(key, field, value)
        await cast(Awaitable[bool], self._redis.expire(key, MAX_TURN_LIFETIME_SECONDS))

    async def admit(self, envelope: TurnEnvelope) -> None:
        key = self.key(envelope.tree_id)
        ceiling, spent, max_nodes = await cast(
            Awaitable[list[str | None]],
            self._redis.hmget(key, ["ceiling", "spent", "max_nodes"]),
        )
        if ceiling is None or spent is None or max_nodes is None:
            raise TreeRefusal("This task's tree has closed; nothing more can start.")
        # A root is gated by the per-user rate limit before it gets here; the
        # spend check is for what the root spawns.
        if envelope.depth > 0 and int(spent) >= int(ceiling):
            raise TreeRefusal(
                "This task has spent its budget; report what you have instead "
                "of starting more work."
            )
        nodes = await self._hincrby(key, "nodes", 1)
        if nodes > int(max_nodes):
            await self._hincrby(key, "nodes", -1)
            raise TreeRefusal(
                f"This task already has {max_nodes} agents working on it; wait "
                "for one to finish or do the work yourself."
            )

    async def release(self, tree_id: str) -> None:
        """Undo an admit whose dispatch never happened."""
        await self._hincrby(self.key(tree_id), "nodes", -1)

    async def charge(self, tree_id: str, microdollars: int) -> None:
        if microdollars <= 0:
            return
        key = self.key(tree_id)
        if not await cast(Awaitable[bool], self._redis.hexists(key, "ceiling")):
            return
        await self._hincrby(key, "spent", microdollars)

    async def snapshot(self, tree_id: str) -> dict[str, int]:
        raw = await cast(
            Awaitable[dict[str, str]], self._redis.hgetall(self.key(tree_id))
        )
        return {k: int(v) for k, v in raw.items()}

    # redis-py types the async client's hash commands as ``Awaitable[T] | T``;
    # the narrowing cast is the pattern the rest of the backend uses.
    async def _hsetnx(self, key: str, field: str, value: int) -> None:
        await cast(Awaitable[bool], self._redis.hsetnx(key, field, str(value)))

    async def _hincrby(self, key: str, field: str, amount: int) -> int:
        return await cast(Awaitable[int], self._redis.hincrby(key, field, amount))


async def get_tree_ledger() -> TreeLedger:
    return TreeLedger(await get_redis_async())


async def resolve_root_ceiling_microdollars(user_id: str | None) -> int:
    """A root tree may spend the configured tree ceiling or whatever the
    user has left this day/week, whichever is smaller."""
    static = config.tree_ceiling_microdollars
    if not user_id:
        return static
    daily, weekly, _ = await get_global_rate_limits(
        user_id,
        config.daily_cost_limit_microdollars,
        config.weekly_cost_limit_microdollars,
    )
    remaining_usd = await get_remaining_usd_budget(
        user_id=user_id, daily_cost_limit=daily, weekly_cost_limit=weekly, floor_usd=0.0
    )
    if remaining_usd == float("inf"):
        return static
    return max(0, min(static, int(round(remaining_usd * 1_000_000))))


async def admit_turn(
    envelope: TurnEnvelope, *, user_id: str | None, ledger: TreeLedger | None = None
) -> None:
    """Open the tree for a root, then admit. Spawned turns fail closed when
    the ledger is unreachable; a root runs as it does today, since the
    per-user rate limit already gates it."""
    try:
        ledger = ledger or await get_tree_ledger()
        if envelope.depth == 0:
            await ledger.open(
                envelope.tree_id,
                ceiling_microdollars=await resolve_root_ceiling_microdollars(user_id),
                max_nodes=config.tree_max_nodes,
            )
        await ledger.admit(envelope)
    except TreeRefusal:
        raise
    except Exception as e:
        if envelope.depth > 0:
            logger.warning(f"Tree ledger unavailable; refusing spawn: {e}")
            raise TreeRefusal(
                "Could not account for this work right now; try again shortly."
            ) from e
        logger.warning(f"Tree ledger unavailable for root turn; continuing: {e}")


async def release_turn(envelope: TurnEnvelope) -> None:
    try:
        await (await get_tree_ledger()).release(envelope.tree_id)
    except Exception as e:
        logger.warning(f"Tree ledger release failed for {envelope.tree_id}: {e}")


async def charge_turn(envelope: TurnEnvelope, microdollars: int) -> None:
    try:
        await (await get_tree_ledger()).charge(envelope.tree_id, microdollars)
    except Exception as e:
        logger.warning(f"Tree ledger charge failed for {envelope.tree_id}: {e}")
