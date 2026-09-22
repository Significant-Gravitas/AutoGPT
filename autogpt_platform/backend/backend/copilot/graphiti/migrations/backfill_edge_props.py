"""Backfill custom edge properties on existing ``:RELATES_TO`` edges.

Created 2026-05-12 alongside the introduction of custom Graphiti entity
and edge types (``graphiti/types.py``). Before custom types were wired
into ``add_episode``, the LLM extractor produced edges with only
``fact``/``name``/temporal-validity properties. The dream pass and
ratification loop want to filter edges by ``status`` natively in Cypher,
which requires the property to exist on every edge.

This script walks every per-user FalkorDB database and stamps
``status`` / ``source_kind`` / ``scope`` on every ``:RELATES_TO`` edge
where they are unset. It is idempotent — re-running is a no-op.

``source_kind`` and ``scope`` are NOT blanket defaults: the original
``MemoryEnvelope`` JSON survives in ``:Episodic.content``, so the
migration joins each edge back to its source episodes and recovers the
envelope's real values. Without this, a pre-fix edge whose envelope was
``assistant_derived`` (a dream write) would be branded
``user_asserted`` — exactly the provenance signal ratification uses to
distinguish dream facts from user facts. Defaults apply only when no
envelope evidence exists (conversation-derived edges, missing
episodes).

Usage:

    poetry run python -m \\
        backend.copilot.graphiti.migrations.backfill_edge_props

By default it processes every user. Pass ``--user-id <id>`` to backfill
one user (useful for canary / debugging).
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict

from backend.copilot.graphiti.client import derive_group_id
from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver
from backend.copilot.graphiti.memory_model import MemoryEnvelope, SourceKind

logger = logging.getLogger(__name__)


DEFAULT_SOURCE_KIND = SourceKind.user_asserted.value
DEFAULT_SCOPE = "real:global"

# When an edge merges facts from multiple episodes (graphiti dedup), the
# most-trusted source among them wins: a fact the user actually asserted
# stays user-trusted even if a dream write later merged into the same
# edge. Conversation-derived (non-envelope) episodes count as
# ``user_asserted`` — that was the implicit assumption before envelopes
# carried explicit metadata.
_SOURCE_KIND_PRECEDENCE = (
    SourceKind.user_asserted,
    SourceKind.tool_observed,
    SourceKind.assistant_derived,
)

SELECT_NULL_EDGES_QUERY = """
MATCH ()-[e:RELATES_TO]->()
WHERE e.status IS NULL OR e.source_kind IS NULL OR e.scope IS NULL
RETURN e.uuid AS uuid, e.episodes AS episodes
"""

EPISODE_CONTENT_QUERY = """
MATCH (ep:Episodic)
WHERE ep.uuid IN $uuids
RETURN ep.uuid AS uuid, ep.content AS content
"""

# Each property is ``coalesce``-guarded so the query is idempotent AND a
# partial prior run (or a real value already set by
# ``mark_edges_superseded``) is never clobbered.
#
# ``status`` is temporal-aware, gated on ``expired_at`` ONLY: graphiti
# stamps ``expired_at = now()`` exactly when an edge is actually
# invalidated, so it's the reliable "already retired" signal — such an
# edge defaults to ``'superseded'`` rather than being mislabeled
# ``'active'``. We deliberately do NOT trigger on ``invalid_at``:
# ``invalid_at`` is parsed from LLM-extracted dates and can be
# FUTURE-dated (a fact true now but with a known end date), so a
# currently-valid edge would otherwise be wrongly marked superseded.
#
# ``status`` is also deliberately NOT recovered from the envelope: a
# pre-fix dream proposal whose envelope said ``tentative`` has been
# behaving as active ever since, and resurrecting ``tentative`` now
# would feed ratification edges with zero recorded hits and >30d age —
# mass supersession on the next pass. ``confidence``/``provenance``
# legitimately default to NULL, so we leave them unset.
APPLY_DEFAULTS_QUERY = """
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid IN $uuids
SET e.source_kind = coalesce(e.source_kind, $source_kind),
    e.scope = coalesce(e.scope, $scope),
    e.status = coalesce(
        e.status,
        CASE WHEN e.expired_at IS NOT NULL THEN 'superseded' ELSE 'active' END
    )
RETURN count(e) AS updated
"""

# Page size for the Postgres User scan. We process users in fixed-size
# batches ordered by ``id`` so the migration's memory footprint stays
# constant even when the user base grows to millions of rows.
USER_BATCH_SIZE = 1000

# Max uuids per ``IN $uuids`` Cypher parameter list.
_UUID_CHUNK_SIZE = 500


async def backfill_one_user(user_id: str) -> int:
    """Recover/default edge metadata for one user. Returns count updated."""
    try:
        group_id = derive_group_id(user_id)
    except ValueError:
        logger.warning(
            "Skipping user %s — invalid for group_id derivation", user_id[:12]
        )
        return 0

    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
    )
    try:
        updated = await backfill_graph(driver)
        if updated:
            logger.info("Backfilled %d edges for user %s", updated, user_id[:12])
        return updated
    except Exception:
        # User may have no graph yet (never used memory). Treat as no-op.
        logger.debug(
            "No graph found or query failed for user %s", user_id[:12], exc_info=True
        )
        return 0
    finally:
        await driver.close()


async def backfill_graph(driver: AutoGPTFalkorDriver) -> int:
    """Backfill every NULL-metadata edge in one graph. Returns count updated.

    Three steps: select the edges missing metadata, recover each edge's
    ``source_kind``/``scope`` from its source-episode envelopes, then
    apply one coalesce-guarded SET per distinct (source_kind, scope)
    bucket.
    """
    edges = await _fetch_null_edges(driver)
    if not edges:
        return 0

    episode_uuids = {uuid for _, uuids in edges for uuid in uuids}
    envelopes = await _fetch_envelopes(driver, sorted(episode_uuids))

    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for edge_uuid, edge_episode_uuids in edges:
        recovered = _recover_edge_metadata(edge_episode_uuids, envelopes)
        buckets[recovered].append(edge_uuid)

    updated = 0
    for (source_kind, scope), edge_uuids in buckets.items():
        for chunk in _chunked(edge_uuids):
            result = await driver.execute_query(
                APPLY_DEFAULTS_QUERY,
                uuids=chunk,
                source_kind=source_kind,
                scope=scope,
            )
            records = result[0] if result else []
            updated += records[0]["updated"] if records else 0
    return updated


def _recover_edge_metadata(
    episode_uuids: list[str],
    envelopes: dict[str, MemoryEnvelope],
) -> tuple[str, str]:
    """Resolve (source_kind, scope) for an edge from its source episodes.

    An episode uuid absent from ``envelopes`` is a non-envelope source
    (conversation turn, or an episode node that no longer exists) and
    votes ``user_asserted`` / ``real:global``. With no episodes at all
    the historical defaults apply unchanged.
    """
    parsed = [envelopes[uuid] for uuid in episode_uuids if uuid in envelopes]
    if not parsed:
        return DEFAULT_SOURCE_KIND, DEFAULT_SCOPE

    has_non_envelope = len(parsed) < len(episode_uuids)
    kinds = {envelope.source_kind for envelope in parsed}
    if has_non_envelope:
        kinds.add(SourceKind.user_asserted)
    source_kind = next(k for k in _SOURCE_KIND_PRECEDENCE if k in kinds).value

    scopes = {envelope.scope for envelope in parsed}
    if has_non_envelope:
        scopes.add(DEFAULT_SCOPE)
    scope = next(iter(scopes)) if len(scopes) == 1 else DEFAULT_SCOPE
    return source_kind, scope


async def _fetch_null_edges(
    driver: AutoGPTFalkorDriver,
) -> list[tuple[str, list[str]]]:
    """Return (edge_uuid, episode_uuids) for every edge missing metadata."""
    result = await driver.execute_query(SELECT_NULL_EDGES_QUERY)
    records = result[0] if result else []
    return [(r["uuid"], r["episodes"] or []) for r in records]


async def _fetch_envelopes(
    driver: AutoGPTFalkorDriver, episode_uuids: list[str]
) -> dict[str, MemoryEnvelope]:
    """Fetch episode contents and parse the ones that are envelopes.

    Episodes whose content is not valid ``MemoryEnvelope`` JSON
    (conversation turns are plain text) are simply omitted.
    """
    envelopes: dict[str, MemoryEnvelope] = {}
    for chunk in _chunked(episode_uuids):
        result = await driver.execute_query(EPISODE_CONTENT_QUERY, uuids=chunk)
        records = result[0] if result else []
        for record in records:
            envelope = _parse_envelope(record["content"])
            if envelope is not None:
                envelopes[record["uuid"]] = envelope
    return envelopes


def _parse_envelope(content: object) -> MemoryEnvelope | None:
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return MemoryEnvelope.model_validate(payload)
    except ValueError:
        return None


def _chunked(items: list[str]) -> list[list[str]]:
    return [
        items[i : i + _UUID_CHUNK_SIZE] for i in range(0, len(items), _UUID_CHUNK_SIZE)
    ]


async def backfill_all_users() -> tuple[int, int]:
    """Walk every User row in Postgres and backfill their graph.

    Pages through ``User`` ordered by ``id`` so the migration's memory
    footprint stays bounded by ``USER_BATCH_SIZE`` regardless of how
    large the user table grows. Each page is fetched with
    ``id > last_seen_id`` (keyset / cursor pagination), which is
    O(log n) per page on the primary-key index — no growing OFFSET cost.

    Returns (users_processed, edges_updated).
    """
    from prisma import Prisma

    db = Prisma()
    await db.connect()
    total_users = 0
    total_edges = 0
    try:
        last_seen_id: str | None = None
        while True:
            where = {"id": {"gt": last_seen_id}} if last_seen_id else {}
            batch = await db.user.find_many(
                where=where,
                order={"id": "asc"},
                take=USER_BATCH_SIZE,
            )
            if not batch:
                break
            for user in batch:
                total_users += 1
                updated = await backfill_one_user(user.id)
                total_edges += updated
            last_seen_id = batch[-1].id
            if len(batch) < USER_BATCH_SIZE:
                break
    finally:
        await db.disconnect()

    return total_users, total_edges


async def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.user_id:
        updated = await backfill_one_user(args.user_id)
        print(f"backfilled {updated} edges for user {args.user_id[:12]}")
        return 0

    users, edges = await backfill_all_users()
    print(f"backfilled {edges} edges across {users} users")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user-id",
        help="Backfill a single user instead of all users. Useful for canary runs.",
    )
    sys.exit(asyncio.run(main(parser.parse_args())))
