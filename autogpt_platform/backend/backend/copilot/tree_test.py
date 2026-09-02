"""Tests for the per-tree bounds on spawned copilot turns.

The envelope derivation is pure and is tested for the property that matters:
every field can only narrow on descent. The ledger is tested against a
minimal in-memory Redis that mirrors the hash operations it uses.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from backend.copilot.permissions import ALL_TOOL_NAMES, CopilotPermissions
from backend.data.redis_client import AsyncRedisClient

from . import tree
from .tree import (
    DESCENT_DENIED_TOOLS,
    ISOLATE_DENIED_TOOLS,
    MAX_DEPTH,
    SPAWN_TOOLS,
    SpawnRequest,
    TreeLedger,
    TreeRefusal,
    TurnEnvelope,
    admit_turn,
    derive_child_envelope,
    root_envelope,
)


class FakeRedis:
    """Just the hash operations ``TreeLedger`` uses, with redis-py semantics
    under ``decode_responses=True``."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.ttls: dict[str, int] = {}

    async def hsetnx(self, key: str, field: str, value: Any) -> int:
        bucket = self.hashes.setdefault(key, {})
        if field in bucket:
            return 0
        bucket[field] = str(value)
        return 1

    async def hmget(self, key: str, fields: list[str]) -> list[str | None]:
        bucket = self.hashes.get(key, {})
        return [bucket.get(f) for f in fields]

    async def hincrby(self, key: str, field: str, amount: int) -> int:
        bucket = self.hashes.setdefault(key, {})
        value = int(bucket.get(field, "0")) + amount
        bucket[field] = str(value)
        return value

    async def hexists(self, key: str, field: str) -> bool:
        return field in self.hashes.get(key, {})

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def expire(self, key: str, seconds: int) -> int:
        self.ttls[key] = seconds
        return 1


class BrokenRedis:
    def __getattr__(self, name: str):
        async def _fail(*_a: Any, **_k: Any) -> None:
            raise ConnectionError("redis down")

        return _fail


def _child(spawner: TurnEnvelope, **request: Any) -> TurnEnvelope:
    return derive_child_envelope(spawner, SpawnRequest(**request))


# ── derivation ─────────────────────────────────────────────────────────


def test_root_is_unrestricted_and_depth_zero() -> None:
    root = root_envelope("turn-1")
    assert root.depth == 0
    assert root.tools is None
    assert root.permits("run_agent")
    assert root.as_permissions() is None


def test_delegate_default_drops_only_account_tools_keeps_its_job() -> None:
    # A delegate runs under its own identity: it keeps the tools it works with.
    child = _child(root_envelope("t"))
    assert child.depth == 1
    assert child.tools is not None
    assert child.tools.isdisjoint(DESCENT_DENIED_TOOLS)
    assert child.tools.isdisjoint(SPAWN_TOOLS)
    assert "read_workspace_file" in child.tools
    assert ISOLATE_DENIED_TOOLS <= child.tools  # post/schedule/memory_store kept
    assert child.as_permissions() == CopilotPermissions(
        tools=sorted(child.tools), tools_exclude=False
    )


def test_isolate_also_drops_outward_and_memory_write_tools() -> None:
    isolate = _child(root_envelope("t"), shares_memory=True, may_spawn=True)
    assert isolate.tools is not None
    assert isolate.tools.isdisjoint(ISOLATE_DENIED_TOOLS)
    assert isolate.tools.isdisjoint(DESCENT_DENIED_TOOLS)
    assert "memory_search" in isolate.tools  # reading is fine; writing is not


def test_grant_cannot_reach_a_tool_denied_in_this_context() -> None:
    """An explicit denial is absolute — grant is not a way around it."""
    isolate = _child(
        root_envelope("t"), shares_memory=True, grant=["post_to_chat_platform"]
    )
    assert isolate.tools is not None
    assert "post_to_chat_platform" not in isolate.tools  # isolate-denied
    assert "connect_integration" not in isolate.tools  # descent-denied
    # The grant is intersected like everything else: a spawner without the
    # tool cannot hand it down.
    narrow_parent = _child(
        root_envelope("t"),
        tools=["read_workspace_file", "run_sub_session"],
        may_spawn=True,
    )
    grandchild = derive_child_envelope(
        narrow_parent, SpawnRequest(grant=["post_to_chat_platform"])
    )
    assert grandchild.tools == frozenset({"read_workspace_file"})


def test_grant_adds_a_non_denied_tool_the_spawner_holds() -> None:
    """Grant still works for anything not on a denial list."""
    delegate = _child(root_envelope("t"), grant=["post_to_chat_platform"])
    assert delegate.tools is not None
    assert "post_to_chat_platform" in delegate.tools


def test_may_spawn_keeps_spawn_tools_but_nothing_denied() -> None:
    child = _child(root_envelope("t"), may_spawn=True)
    assert child.tools is not None
    assert SPAWN_TOOLS <= child.tools
    assert child.tools.isdisjoint(DESCENT_DENIED_TOOLS)


def test_pinning_a_tool_set_still_respects_the_denials() -> None:
    child = _child(root_envelope("t"), tools=["post_to_chat_platform"])
    assert child.tools == frozenset({"post_to_chat_platform"})
    # …but a descent-denied tool is refused even when pinned by name.
    denied = _child(root_envelope("t"), tools=["connect_integration"])
    assert denied.tools == frozenset()


def test_child_cannot_regain_a_tool_its_spawner_lacks() -> None:
    parent = _child(
        root_envelope("t"),
        tools=["read_workspace_file", "run_sub_session"],
        may_spawn=True,
    )
    assert parent.tools == frozenset({"read_workspace_file", "run_sub_session"})
    grandchild = derive_child_envelope(
        parent, SpawnRequest(tools=["read_workspace_file", "run_agent", "bash_exec"])
    )
    assert grandchild.tools == frozenset({"read_workspace_file"})


def test_leaf_cannot_spawn() -> None:
    leaf = _child(root_envelope("t"))
    with pytest.raises(TreeRefusal):
        derive_child_envelope(leaf, SpawnRequest())


def test_depth_is_bounded_for_every_spawn_kind() -> None:
    node = root_envelope("t")
    for _ in range(MAX_DEPTH):
        node = derive_child_envelope(node, SpawnRequest(may_spawn=True))
    assert node.depth == MAX_DEPTH
    with pytest.raises(TreeRefusal):
        derive_child_envelope(node, SpawnRequest(may_spawn=True))


def test_spawner_permissions_narrow_the_ceiling() -> None:
    perms = CopilotPermissions(
        tools=["read_workspace_file", "web_fetch"], tools_exclude=False
    )
    child = derive_child_envelope(
        root_envelope("t"),
        SpawnRequest(tools=["web_fetch", "bash_exec"]),
        spawner_permissions=perms,
    )
    assert child.tools == frozenset({"web_fetch"})


def test_taint_only_ever_rises() -> None:
    clean = root_envelope("t")
    assert not _child(clean).tainted
    assert _child(clean, born_tainted=True).tainted
    tainted = root_envelope("t", tainted=True)
    assert _child(tainted, born_tainted=False).tainted


def test_deadline_is_the_minimum() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    parent = derive_child_envelope(
        root_envelope("t"), SpawnRequest(max_seconds=600, may_spawn=True), now=now
    )
    assert parent.deadline_at == now + timedelta(seconds=600)
    child = derive_child_envelope(parent, SpawnRequest(max_seconds=3600), now=now)
    assert child.deadline_at == parent.deadline_at
    later = derive_child_envelope(parent, SpawnRequest(max_seconds=60), now=now)
    assert later.deadline_at == now + timedelta(seconds=60)


def test_expired_spawner_cannot_spawn() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    parent = derive_child_envelope(
        root_envelope("t"), SpawnRequest(max_seconds=10, may_spawn=True), now=now
    )
    with pytest.raises(TreeRefusal):
        derive_child_envelope(parent, SpawnRequest(), now=now + timedelta(seconds=11))


def test_tree_id_and_tenancy_free_fields_never_change() -> None:
    root = root_envelope("turn-9")
    node = root
    for _ in range(MAX_DEPTH):
        node = derive_child_envelope(node, SpawnRequest(may_spawn=True))
        assert node.tree_id == "turn-9"


def test_envelope_round_trips_through_the_queue() -> None:
    child = _child(root_envelope("t"), tools=["read_workspace_file"])
    restored = TurnEnvelope.model_validate_json(child.model_dump_json())
    assert restored == child
    assert isinstance(restored.tools, frozenset)


# ── ledger ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_root_opens_tree_and_children_count_against_it() -> None:
    redis = FakeRedis()
    ledger = TreeLedger(cast(AsyncRedisClient, redis))
    root = root_envelope("t")
    await ledger.open("t", ceiling_microdollars=1_000_000, max_nodes=3)
    await ledger.admit(root)
    await ledger.admit(_child(root))
    await ledger.admit(_child(root))
    with pytest.raises(TreeRefusal):
        await ledger.admit(_child(root))
    assert (await ledger.snapshot("t"))["nodes"] == 3


@pytest.mark.asyncio
async def test_over_admit_is_rolled_back() -> None:
    redis = FakeRedis()
    ledger = TreeLedger(cast(AsyncRedisClient, redis))
    await ledger.open("t", ceiling_microdollars=1_000_000, max_nodes=1)
    await ledger.admit(root_envelope("t"))
    with pytest.raises(TreeRefusal):
        await ledger.admit(_child(root_envelope("t")))
    assert (await ledger.snapshot("t"))["nodes"] == 1
    await ledger.release("t")
    assert (await ledger.snapshot("t"))["nodes"] == 0


@pytest.mark.asyncio
async def test_concurrent_admits_never_exceed_the_cap() -> None:
    redis = FakeRedis()
    ledger = TreeLedger(cast(AsyncRedisClient, redis))
    await ledger.open("t", ceiling_microdollars=1_000_000, max_nodes=8)
    root = root_envelope("t")
    await ledger.admit(root)

    async def try_admit() -> bool:
        try:
            await ledger.admit(_child(root))
            return True
        except TreeRefusal:
            return False

    results = await asyncio.gather(*(try_admit() for _ in range(30)))
    assert sum(results) == 7
    assert (await ledger.snapshot("t"))["nodes"] == 8


@pytest.mark.asyncio
async def test_spend_ceiling_refuses_children_but_not_the_root() -> None:
    redis = FakeRedis()
    ledger = TreeLedger(cast(AsyncRedisClient, redis))
    await ledger.open("t", ceiling_microdollars=500_000, max_nodes=10)
    root = root_envelope("t")
    await ledger.admit(root)
    await ledger.charge("t", 300_000)
    await ledger.admit(_child(root))
    await ledger.charge("t", 300_000)
    with pytest.raises(TreeRefusal):
        await ledger.admit(_child(root))
    assert (await ledger.snapshot("t"))["spent"] == 600_000


@pytest.mark.asyncio
async def test_charge_to_unknown_tree_is_ignored() -> None:
    redis = FakeRedis()
    await TreeLedger(cast(AsyncRedisClient, redis)).charge("ghost", 1)
    assert "copilot:tree:ghost" not in redis.hashes


@pytest.mark.asyncio
async def test_child_of_a_closed_tree_is_refused() -> None:
    with pytest.raises(TreeRefusal):
        await TreeLedger(cast(AsyncRedisClient, FakeRedis())).admit(
            _child(root_envelope("gone"))
        )


@pytest.mark.asyncio
async def test_admit_turn_fails_closed_for_children_open_for_roots(monkeypatch) -> None:
    async def _ceiling(_user_id):
        return 1_000_000

    monkeypatch.setattr(tree, "resolve_root_ceiling_microdollars", _ceiling)
    broken = TreeLedger(cast(AsyncRedisClient, BrokenRedis()))
    # A root never touches the ledger, so a dead Redis cannot block a chat.
    await admit_turn(root_envelope("t"), user_id="u", ledger=broken)
    with pytest.raises(TreeRefusal):
        await admit_turn(_child(root_envelope("t")), user_id="u", ledger=broken)


@pytest.mark.asyncio
async def test_first_spawn_opens_the_tree_with_the_root_counted(monkeypatch) -> None:
    async def _ceiling(_user_id):
        return 42

    monkeypatch.setattr(tree, "resolve_root_ceiling_microdollars", _ceiling)
    monkeypatch.setattr(tree.config, "tree_max_nodes", 2)
    redis = FakeRedis()
    ledger = TreeLedger(cast(AsyncRedisClient, redis))
    root = root_envelope("t")
    await admit_turn(root, user_id="u", ledger=ledger)
    assert "copilot:tree:t" not in redis.hashes
    await admit_turn(_child(root), user_id="u", ledger=ledger)
    snapshot = await ledger.snapshot("t")
    assert snapshot == {"ceiling": 42, "spent": 0, "nodes": 2, "max_nodes": 2}
    assert redis.ttls["copilot:tree:t"] > 0
    with pytest.raises(TreeRefusal):
        await admit_turn(_child(root), user_id="u", ledger=ledger)


def test_all_tool_names_cover_the_denied_set() -> None:
    # A denied name that is not a real tool would be a silent no-op.
    assert DESCENT_DENIED_TOOLS <= ALL_TOOL_NAMES
    assert SPAWN_TOOLS <= ALL_TOOL_NAMES


# ── ceiling formula (finding 3) ────────────────────────────────────────


def _ceiling(daily: int, remaining_usd: float, monkeypatch) -> int:
    """Drive resolve_root_ceiling_microdollars with a fixed tier + balance."""

    async def _limits(_uid, _d, _w):
        return daily, daily * 5, "TIER"

    async def _remaining(**_kw):
        return remaining_usd

    monkeypatch.setattr(tree, "get_global_rate_limits", _limits)
    monkeypatch.setattr(tree, "get_remaining_usd_budget", _remaining)
    return asyncio.run(tree.resolve_root_ceiling_microdollars("u"))


def test_ceiling_is_a_fraction_of_the_tier_daily_limit(monkeypatch) -> None:
    monkeypatch.setattr(tree.config, "tree_ceiling_fraction_of_daily", 0.5)
    monkeypatch.setattr(tree.config, "tree_ceiling_floor_microdollars", 500_000)
    monkeypatch.setattr(tree.config, "tree_ceiling_microdollars", 10_000_000)
    # 0.5 x $8.00 = $4.00, under both floor and cap.
    assert _ceiling(8_000_000, 999.0, monkeypatch) == 4_000_000


def test_ceiling_floor_applies_to_a_small_tier(monkeypatch) -> None:
    monkeypatch.setattr(tree.config, "tree_ceiling_fraction_of_daily", 0.5)
    monkeypatch.setattr(tree.config, "tree_ceiling_floor_microdollars", 500_000)
    monkeypatch.setattr(tree.config, "tree_ceiling_microdollars", 10_000_000)
    # 0.5 x $0.20 = $0.10, raised to the $0.50 floor so a tree can fund a turn.
    assert _ceiling(200_000, 999.0, monkeypatch) == 500_000


def test_ceiling_cap_applies_to_a_large_tier(monkeypatch) -> None:
    monkeypatch.setattr(tree.config, "tree_ceiling_fraction_of_daily", 0.5)
    monkeypatch.setattr(tree.config, "tree_ceiling_floor_microdollars", 500_000)
    monkeypatch.setattr(tree.config, "tree_ceiling_microdollars", 10_000_000)
    assert _ceiling(400_000_000, 999.0, monkeypatch) == 10_000_000


def test_no_tier_yields_no_budget_and_therefore_no_spawns(monkeypatch) -> None:
    """A NO_TIER user's multiplier is 0.0, so the daily limit resolves to 0.
    The floor must not resurrect a tier that may not spend at all."""
    monkeypatch.setattr(tree.config, "tree_ceiling_floor_microdollars", 500_000)
    assert _ceiling(0, 0.0, monkeypatch) == 0


def test_remaining_budget_clamps_the_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(tree.config, "tree_ceiling_fraction_of_daily", 0.5)
    monkeypatch.setattr(tree.config, "tree_ceiling_floor_microdollars", 500_000)
    monkeypatch.setattr(tree.config, "tree_ceiling_microdollars", 10_000_000)
    # Formula would allow $4.00; only $0.75 is left today.
    assert _ceiling(8_000_000, 0.75, monkeypatch) == 750_000


def test_isolate_denied_names_are_real_tools() -> None:
    # DESCENT and SPAWN are asserted above; ISOLATE was the gap.
    assert ISOLATE_DENIED_TOOLS <= ALL_TOOL_NAMES


# ── no amplification (finding 4) ───────────────────────────────────────


def _spawners_lacking(tool: str) -> list[tuple[str, TurnEnvelope]]:
    """Every shape of spawner that does not hold *tool*.

    Which shapes qualify depends on the tool: a delegate keeps the
    isolate-denied set, so it is only a valid "lacking" spawner for a
    descent-denied tool.
    """
    root = root_envelope("t")
    candidates = [
        (
            "explicit pin",
            derive_child_envelope(
                root,
                SpawnRequest(
                    tools=["read_workspace_file", "run_sub_session"], may_spawn=True
                ),
            ),
        ),
        (
            "isolate default",
            derive_child_envelope(
                root, SpawnRequest(shares_memory=True, may_spawn=True)
            ),
        ),
        (
            "delegate default",
            derive_child_envelope(root, SpawnRequest(may_spawn=True)),
        ),
    ]
    lacking = [(label, env) for label, env in candidates if not env.permits(tool)]
    assert lacking, f"no spawner shape lacks {tool}"
    return lacking


@pytest.mark.parametrize("tool", ["post_to_chat_platform", "connect_integration"])
def test_a_spawner_can_never_grant_a_tool_it_does_not_hold(tool: str) -> None:
    """An agent cannot authorize use of a tool it is not itself authorized to
    use — for every combination of the request's tool inputs."""
    requests = [
        SpawnRequest(grant=[tool], may_spawn=True),
        SpawnRequest(tools=[tool], may_spawn=True),
        SpawnRequest(tools=[tool], grant=[tool], may_spawn=True),
        SpawnRequest(tools=None, grant=[tool, "bash_exec"], may_spawn=True),
        SpawnRequest(tools=[tool, "read_workspace_file"], grant=[tool]),
    ]
    for label, spawner in _spawners_lacking(tool):
        assert not spawner.permits(tool), label
        for request in requests:
            child = derive_child_envelope(spawner, request)
            assert not child.permits(tool), f"{label} amplified via {request}"
            assert child.tools is not None
            assert child.tools <= (spawner.tools or ALL_TOOL_NAMES), label


@pytest.mark.parametrize("tool", sorted(DESCENT_DENIED_TOOLS))
def test_no_spawner_can_ever_pass_on_an_explicitly_denied_tool(tool: str) -> None:
    """Even a spawner that holds the tool — a root holds everything — cannot
    hand a descent-denied tool to a child, by any mechanism."""
    root = root_envelope("t")
    assert root.permits(tool)
    for request in (
        SpawnRequest(grant=[tool], may_spawn=True),
        SpawnRequest(tools=[tool], may_spawn=True),
        SpawnRequest(tools=[tool], grant=[tool], may_spawn=True),
        SpawnRequest(tools=None, grant=[tool], shares_memory=True),
    ):
        assert not derive_child_envelope(root, request).permits(tool)


def test_a_spawner_may_still_grant_a_non_denied_tool() -> None:
    root = root_envelope("t")
    child = derive_child_envelope(root, SpawnRequest(grant=["post_to_chat_platform"]))
    assert child.permits("post_to_chat_platform")
    assert not child.permits("delete_preset")
