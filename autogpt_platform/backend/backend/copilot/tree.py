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

What that costs, stated accurately: admission reads spend recorded by turns
that have *finished*, so turns admitted concurrently do not see each other's
cost. A tree can therefore overshoot its ceiling by up to ``max_nodes``
in-flight turns, not by one. The node cap is what bounds it — which is the
reason the two limits are enforced together and why ``max_nodes`` is small.

Reachable surface, stated plainly so nobody reads more into this module than
it currently does. Live today: ``depth``, ``tools`` via the descent/isolate
defaults and ``grant``, the node cap and the spend ceiling. Carried but not
yet driven by any tool: ``tainted`` (propagated and tested, but no consumer
reads it until the auto-mode gate lands), ``SpawnRequest.max_seconds`` (so
``deadline_at`` is always ``None``), ``SpawnRequest.tools`` (the exact-set pin
— which means the read-only quarantine shape is expressible here but not
selectable from any tool yet), and ``may_spawn=False`` (all three spawn tools
pass ``True``, so no leaf is created in production).
"""

import logging
from collections.abc import Awaitable
from datetime import UTC, datetime, timedelta
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

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

# Denied to EVERY spawned child by default: irreversible deletes, and actions
# that bind the user's account (connect an integration, register a trigger,
# staff the team). None is part of a delegated expert's normal job, so
# withholding them does not break a working delegation — it removes reach a
# spawned turn should never have silently. The staffing tools are already
# denied to expert sessions by their tool group; listing them keeps the
# guarantee even for a plain-AutoPilot child.
DESCENT_DENIED_TOOLS: frozenset[str] = frozenset(
    {
        "connect_integration",
        "setup_agent_webhook_trigger",
        "update_preset",
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

# Additionally denied to an isolate — a child that shares its spawner's memory
# namespace and identity (``run_sub_session``). It is scratch space: posting
# outward, scheduling, calling an MCP server, or persisting a skill under the
# parent's identity is not its role, and a memory write it makes is read back
# by the parent next turn as the parent's own. A *delegated* expert runs under
# its own identity, namespace and budget and keeps all of these — its job is
# to do real work.
ISOLATE_DENIED_TOOLS: frozenset[str] = frozenset(
    {
        "post_to_chat_platform",
        "schedule_followup",
        "run_mcp_tool",
        "store_skill",
        "memory_store",
        "add_understanding",
    }
)

_LEDGER_KEY_PREFIX = "copilot:tree:"

# All-or-nothing tree creation. ``HSETNX`` per field is not equivalent: it
# leaves a window where another caller sees some fields and not others, and
# ``admit`` cannot tell that from a tree that has closed.
_OPEN_TREE_SCRIPT = """
if redis.call("EXISTS", KEYS[1]) == 0 then
    redis.call("HSET", KEYS[1],
        "ceiling", ARGV[1],
        "max_nodes", ARGV[2],
        "nodes", ARGV[3],
        "spent", 0)
    redis.call("EXPIRE", KEYS[1], ARGV[4])
    return 1
end
return 0
"""


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
        if not self.tools:
            # An empty whitelist reads back through effective_allowed_tools as
            # "no filter at all", which would show a fully-locked child every
            # tool. Deny-everything is the honest encoding of the same thing.
            return CopilotPermissions(tools=sorted(ALL_TOOL_NAMES), tools_exclude=True)
        return CopilotPermissions(tools=sorted(self.tools), tools_exclude=False)


class SpawnRequest(BaseModel):
    """What a spawner asks for its child. Every field is clamped, never raised.

    ``extra="forbid"``: on a model whose job is narrowing a security envelope,
    a mistyped field name must fail loudly rather than silently yield the
    widest possible narrowing.
    """

    model_config = ConfigDict(extra="forbid")

    # Exact set (a quarantine preset), or None for the default: everything
    # the spawner holds minus the denial lists.
    #
    # There is deliberately no "grant" escape hatch. Denials are absolute, so
    # anything a grant could legitimately add is already in the default set,
    # and anything it could not add must stay denied — a grant parameter can
    # only ever be a no-op or a violation. A child that genuinely needs a
    # denied tool is a human-approval case.
    tools: list[str] | None = None
    may_spawn: bool = False
    shares_memory: bool = False
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
    """The child's envelope. Raises :class:`TreeRefusal`. Pure given ``now``.

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

    # Two rules, both enforced below, and the result is
    # ``child ⊆ (spawner_effective − denied)``:
    #
    # 1. An explicit descent denial is absolute. Nothing reaches a child that
    #    is on the denied list — not via ``request.tools``, not via ``grant``,
    #    not even from a spawner that holds the tool itself. A child that
    #    genuinely needs one of these is a human-approval case, not a grant.
    # 2. Otherwise no amplification: an agent cannot authorize a tool it is
    #    not itself authorized to use.
    ceiling = spawner.tools if spawner.tools is not None else ALL_TOOL_NAMES
    if spawner_permissions is not None:
        ceiling = ceiling & spawner_permissions.effective_allowed_tools(ALL_TOOL_NAMES)
    denied = DESCENT_DENIED_TOOLS
    if request.shares_memory:
        denied = denied | ISOLATE_DENIED_TOOLS
    requested = (
        frozenset(request.tools)
        if request.tools is not None
        else ALL_TOOL_NAMES - denied
    )
    # ``- denied`` last, so no request shape can route around rule 1 above.
    tools = (ceiling & requested) - denied
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
    one, and that over-admit is the bound's stated slack.

    Spend is metered, not reserved: ``spent`` only advances when a turn ends
    (``token_tracking``), so children dispatched in one round all read the
    same figure and are all admitted. The ceiling therefore bounds how much a
    tree may *start* spending against, not its total — the worst case is
    ``(max_nodes - 1) x`` the per-turn cap, i.e. every remaining node
    admitted at ``spent == 0`` and each running to its own limit. The node
    cap is what keeps that finite. Reserving at admit and reconciling at
    charge is the fix, and needs the SDK to report a per-query spend the
    Codex transport does not currently expose.
    """

    def __init__(self, redis: AsyncRedisClient) -> None:
        self._redis = redis

    @staticmethod
    def key(tree_id: str) -> str:
        return f"{_LEDGER_KEY_PREFIX}{tree_id}"

    async def exists(self, tree_id: str) -> bool:
        return await cast(
            Awaitable[bool], self._redis.hexists(self.key(tree_id), "ceiling")
        )

    async def open(
        self,
        tree_id: str,
        *,
        ceiling_microdollars: int,
        max_nodes: int,
        initial_nodes: int = 0,
    ) -> None:
        """Open the tree, all fields and the TTL or none of them.

        Field-by-field ``HSETNX`` left a window where a second first-child
        could see ``ceiling`` written but ``spent``/``max_nodes`` still
        missing, and ``admit`` reads a partial hash as a closed tree — a
        spurious refusal of a perfectly valid spawn. One script closes it,
        and stays idempotent: the racing loser finds the key populated and
        changes nothing.
        """
        await cast(
            Awaitable[int],
            self._redis.eval(
                _OPEN_TREE_SCRIPT,
                1,
                self.key(tree_id),
                str(max(0, ceiling_microdollars)),
                str(max(1, max_nodes)),
                str(max(0, initial_nodes)),
                str(MAX_TURN_LIFETIME_SECONDS),
            ),
        )

    async def admit(self, envelope: TurnEnvelope) -> None:
        key = self.key(envelope.tree_id)
        ceiling, spent, max_nodes = await cast(
            Awaitable[list[str | None]],
            self._redis.hmget(key, ["ceiling", "spent", "max_nodes"]),
        )
        if ceiling is None or spent is None or max_nodes is None:
            raise TreeRefusal("This task's tree has closed; nothing more can start.")
        # A root is gated by the per-user rate limit before it gets here; the
        # spend check is for what the root spawns. It reads settled spend only,
        # so concurrent admits can overshoot by up to ``max_nodes`` turns — see
        # the module docstring; the node cap is the bound that holds.
        if envelope.depth > 0 and int(spent) >= int(ceiling):
            raise TreeRefusal(_spend_refusal(int(spent), int(ceiling)))
        nodes = await self._hincrby(key, "nodes", 1)
        if nodes > int(max_nodes):
            await self._hincrby(key, "nodes", -1)
            raise TreeRefusal(
                f"This task has already used its {max_nodes} agents — that is a "
                "lifetime budget for this request, not a concurrency limit, so "
                "waiting will not free one. Finish the remaining work yourself."
            )

    async def release(self, tree_id: str) -> None:
        """Undo an admit whose dispatch never happened."""
        if await self.exists(tree_id):
            await self._hincrby(self.key(tree_id), "nodes", -1)

    async def charge(self, tree_id: str, microdollars: int) -> None:
        if microdollars <= 0:
            return
        key = self.key(tree_id)
        if not await cast(Awaitable[bool], self._redis.hexists(key, "ceiling")):
            return
        await self._hincrby(key, "spent", microdollars)

    async def claim_wrapup(self, tree_id: str) -> bool:
        """True for the one turn that first crosses the wrap-up threshold.

        ``HSETNX`` is what makes it once-per-tree under concurrent turns, and
        the ``hexists`` guard keeps it from conjuring a ledger for a tree that
        never spawned anything.
        """
        key = self.key(tree_id)
        if not await cast(Awaitable[bool], self._redis.hexists(key, "ceiling")):
            return False
        return await cast(Awaitable[bool], self._redis.hsetnx(key, "wrapup", "1"))

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
        """Always re-arm the TTL.

        ``HINCRBY`` on a key whose TTL fired in the meantime *recreates* it,
        and only ``open()`` ever issued ``EXPIRE`` — so a charge or release
        landing in that window left a ``copilot:tree:*`` hash with no
        expiry, i.e. a permanent leak. Re-arming is idempotent and costs one
        pipelined command.
        """
        value = await cast(Awaitable[int], self._redis.hincrby(key, field, amount))
        await cast(Awaitable[bool], self._redis.expire(key, MAX_TURN_LIFETIME_SECONDS))
        return value


def _spend_refusal(spent: int, ceiling: int) -> str:
    """Say what is left and what to do instead — an error the model can act on.

    A zero ceiling is not an empty wallet: it is a tier that may not spend at
    all, and telling that model to "wrap up with what you have" hides why.
    """
    if ceiling <= 0:
        return (
            "This account has no subscription, so it cannot start sub-sessions. "
            "Complete the remaining work directly with your tools, or wrap up "
            "with the results you already have. Do not retry."
        )
    return (
        f"Budget limit reached (${spent / 1_000_000:.2f} spent of the "
        f"${ceiling / 1_000_000:.2f} maximum for this task). New sub-sessions "
        "cannot be started. Complete the remaining work directly with your "
        "tools, or wrap up with the results you already have. Do not retry."
    )


async def get_tree_ledger() -> TreeLedger:
    return TreeLedger(await get_redis_async())


async def resolve_root_ceiling_microdollars(user_id: str | None) -> int:
    """What one tree may spend:

        min( remaining_budget,
             max( fraction_of_daily * tier_daily_limit, floor ),
             absolute_cap )

    Scaling off the caller's *tier-scaled* daily limit rather than a flat
    constant is what makes the number proportionate — and it is why a
    NO_TIER user (multiplier 0.0) resolves to 0: a tier that may not spend
    may not spawn either. The floor keeps a modest daily limit from
    producing a tree too small to fund one real turn; the cap keeps a
    generous one from handing a single tree the whole day.
    """
    cap = config.tree_ceiling_microdollars
    if not user_id:
        # Fail closed: without a user there is no tier to scale from, and
        # handing out the full cap is the one fail-open branch this module
        # would otherwise contain.
        return 0
    daily, weekly, _ = await get_global_rate_limits(
        user_id,
        config.daily_cost_limit_microdollars,
        config.weekly_cost_limit_microdollars,
    )
    # ``daily`` is already tier-scaled by get_global_rate_limits.
    scaled = int(config.tree_ceiling_fraction_of_daily * daily)
    # A zero tier allowance means no spend at all; the floor must not
    # resurrect it, so it only applies to a tier that may spend.
    allowance = max(scaled, config.tree_ceiling_floor_microdollars) if daily > 0 else 0
    ceiling = min(allowance, cap)

    remaining_usd = await get_remaining_usd_budget(
        user_id=user_id, daily_cost_limit=daily, weekly_cost_limit=weekly, floor_usd=0.0
    )
    if remaining_usd == float("inf"):
        return max(0, ceiling)
    remaining = int(round(remaining_usd * 1_000_000))
    return max(0, min(remaining, ceiling))


async def admit_turn(
    envelope: TurnEnvelope, *, user_id: str | None, ledger: TreeLedger | None = None
) -> None:
    """Admit a spawned turn against its tree, opening the tree on first use.

    Roots never touch the ledger: the per-user rate limit already gates
    them, and the tree only needs to exist once something is spawned. So
    the HTTP route, the scheduler and ``AutoPilotBlock`` pay nothing here,
    and a spawned turn fails closed when the ledger is unreachable.
    """
    if envelope.depth == 0:
        return
    try:
        ledger = ledger or await get_tree_ledger()
        if not await ledger.exists(envelope.tree_id):
            await ledger.open(
                envelope.tree_id,
                ceiling_microdollars=await resolve_root_ceiling_microdollars(user_id),
                max_nodes=config.tree_max_nodes,
                # The root is a node of its own tree.
                initial_nodes=1,
            )
        await ledger.admit(envelope)
    except TreeRefusal:
        raise
    except Exception as e:
        logger.warning(f"Tree ledger unavailable; refusing spawn: {e}")
        raise TreeRefusal(
            "Could not account for this work right now; try again shortly."
        ) from e


async def release_turn(envelope: TurnEnvelope) -> None:
    if envelope.depth == 0:
        return
    try:
        await (await get_tree_ledger()).release(envelope.tree_id)
    except Exception as e:
        logger.warning(f"Tree ledger release failed for {envelope.tree_id}: {e}")


async def charge_turn(envelope: TurnEnvelope, microdollars: int) -> None:
    """Charge a turn's cost to its tree. A root whose tree never opened (no
    spawns) has nothing to charge, and ``charge`` ignores unknown trees."""
    try:
        await (await get_tree_ledger()).charge(envelope.tree_id, microdollars)
    except Exception as e:
        logger.warning(f"Tree ledger charge failed for {envelope.tree_id}: {e}")
