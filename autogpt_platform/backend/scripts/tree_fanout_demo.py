"""Demonstrate the per-tree bounds on spawned copilot turns, with no LLM.

Builds one root turn and fans out thirty spawn requests through the same
derivation and ledger the executor's chokepoint uses, then asserts the
structural claims from AGENT_COLLABORATION_ARCHITECTURE.md §9:

- every child's tool set is a subset of its spawner's minus the descent-
  denied tools, and a leaf cannot spawn;
- depth never exceeds MAX_DEPTH through an isolate → delegate → isolate chain;
- the (max_nodes + 1)th node is refused, and concurrent admits still respect it;
- once metered spend crosses the ceiling, no further turn in the tree starts.

Run:  poetry run python scripts/tree_fanout_demo.py
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from backend.copilot.tree import (
    DESCENT_DENIED_TOOLS,
    MAX_DEPTH,
    SPAWN_TOOLS,
    SpawnRequest,
    TreeLedger,
    TreeRefusal,
    TurnEnvelope,
    derive_child_envelope,
    root_envelope,
)
from backend.data.redis_client import AsyncRedisClient


class InMemoryRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

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
        bucket[field] = str(int(bucket.get(field, "0")) + amount)
        return int(bucket[field])

    async def hexists(self, key: str, field: str) -> bool:
        return field in self.hashes.get(key, {})

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def expire(self, key: str, seconds: int) -> int:
        return 1

    async def eval(self, script: str, numkeys: int, *args: Any) -> int:
        key = str(args[0])
        ceiling, max_nodes, nodes, _ttl = (str(a) for a in args[1:5])
        if key in self.hashes:
            return 0
        self.hashes[key] = {
            "ceiling": ceiling,
            "max_nodes": max_nodes,
            "nodes": nodes,
            "spent": "0",
        }
        return 1


async def spawn(
    ledger: TreeLedger, spawner: TurnEnvelope, request: SpawnRequest
) -> TurnEnvelope | str:
    try:
        child = derive_child_envelope(spawner, request)
        await ledger.admit(child)
        return child
    except TreeRefusal as refused:
        return refused.message


async def main() -> None:
    ledger = TreeLedger(cast(AsyncRedisClient, InMemoryRedis()))
    root = root_envelope("root-turn")
    await ledger.open("root-turn", ceiling_microdollars=1_000_000, max_nodes=8)
    await ledger.admit(root)
    print(f"root: depth={root.depth} tools=unrestricted ceiling=$1.00 max_nodes=8")

    # 1. Fan out thirty leaves concurrently; only max_nodes - 1 may start.
    quarantine = SpawnRequest(tools=["read_workspace_file"])
    results = await asyncio.gather(
        *(spawn(ledger, root, quarantine) for _ in range(30))
    )
    admitted = [r for r in results if isinstance(r, TurnEnvelope)]
    refused = [r for r in results if isinstance(r, str)]
    print(f"fan-out: {len(admitted)} admitted, {len(refused)} refused")
    assert len(admitted) == 7, len(admitted)
    for leaf in admitted:
        assert leaf.tools == frozenset({"read_workspace_file"})
        assert leaf.depth == 1
        assert not any(leaf.permits(t) for t in SPAWN_TOOLS | DESCENT_DENIED_TOOLS)
    print(
        f"  every leaf: tools={sorted(admitted[0].tools or ())}, cannot spawn, cannot act outward"
    )
    print(f"  8th node refused with: {refused[0]!r}")

    # 2. A leaf cannot spawn, whatever it asks for.
    leaf_attempt = await spawn(ledger, admitted[0], SpawnRequest(tools=["bash_exec"]))
    assert isinstance(leaf_attempt, str)
    print(f"leaf spawn attempt refused: {leaf_attempt!r}")

    # 3. Depth bounds an isolate → delegate → isolate chain even with room.
    deep_ledger = TreeLedger(cast(AsyncRedisClient, InMemoryRedis()))
    await deep_ledger.open("deep", ceiling_microdollars=1_000_000, max_nodes=100)
    node: TurnEnvelope = root_envelope("deep")
    await deep_ledger.admit(node)
    kinds = ["isolate", "delegate", "isolate", "delegate"]
    for hop, kind in enumerate(kinds, start=1):
        outcome = await spawn(deep_ledger, node, SpawnRequest(may_spawn=True))
        if isinstance(outcome, str):
            print(f"hop {hop} ({kind}) refused at depth {node.depth}: {outcome!r}")
            assert node.depth == MAX_DEPTH
            break
        node = outcome
        print(f"hop {hop} ({kind}) admitted at depth {node.depth}")
    else:
        raise AssertionError("depth bound never fired")

    # 4. Default child tools shrink monotonically along the chain.
    default_child = await spawn(
        deep_ledger, root_envelope("deep"), SpawnRequest(may_spawn=True)
    )
    assert isinstance(default_child, TurnEnvelope) and default_child.tools is not None
    assert default_child.tools.isdisjoint(DESCENT_DENIED_TOOLS)
    # connect_integration is descent-denied, so default_child does not hold it
    # and cannot pass it on even when a grandchild asks by name.
    narrower = derive_child_envelope(
        default_child,
        SpawnRequest(tools=["read_workspace_file", "connect_integration"]),
    )
    assert narrower.tools == frozenset({"read_workspace_file"})
    print("a child asking for a tool its spawner lacks does not get it")

    # 5. Metered spend closes the tree.
    spend_ledger = TreeLedger(cast(AsyncRedisClient, InMemoryRedis()))
    await spend_ledger.open("spend", ceiling_microdollars=500_000, max_nodes=50)
    spend_root = root_envelope("spend")
    await spend_ledger.admit(spend_root)
    started = 0
    while True:
        outcome = await spawn(spend_ledger, spend_root, SpawnRequest())
        if isinstance(outcome, str):
            print(f"after {started} charged turns the tree refused: {outcome!r}")
            break
        started += 1
        await spend_ledger.charge("spend", 120_000)
    snapshot = await spend_ledger.snapshot("spend")
    assert snapshot["spent"] >= snapshot["ceiling"]
    print(f"ledger: {snapshot}")
    print("all structural claims hold")


if __name__ == "__main__":
    asyncio.run(main())
