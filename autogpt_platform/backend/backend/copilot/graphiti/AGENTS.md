# Graphiti Memory

This directory contains the Graphiti-backed memory integration for CoPilot.
This file is developer documentation only — it is NOT injected into LLM prompts.
Runtime prompt instructions live in `prompting.py:get_graphiti_supplement()`.

## Scope

- Keep Graphiti and FalkorDB-specific logic in this package.
- Prefer changes here over scattering Graphiti behavior across unrelated copilot modules.

## Debugging

- Use raw FalkorDB queries to inspect stored nodes, episodes, and `RELATES_TO` facts before changing retrieval behavior.
- Distinguish user-provided facts, assistant-generated findings, and provenance/meta entities when evaluating memory quality.

## Design Intent

- Preserve per-user isolation through `group_id`-scoped databases and clients.
- Build a `MemoryScope` (`scope.py`) once where a request enters the memory code and pass it down; read `group_id` / `scope_key` / `redis_key(...)` off it rather than re-deriving them from a `(user_id, expert_id)` pair, and open drivers with `open_driver(scope)`.
- Be careful about memory pollution from assistant/tool phrasing; extraction quality matters as much as ingestion success.
- Keep warm-context and tool-driven recall resilient: failures should degrade gracefully rather than break chat execution.

## Dream Schedules

A memory scope (the account, or one hired expert) that has been registered has one `MemoryScopeSchedule` row, and while that row is ACTIVE, the crons whose flag is on: `dream_nightly_batch_{scope_key}` (daily, 03:00 owner-local) and `community_rebuild_{scope_key}` (weekly, 04:00 owner-local on Mondays: APScheduler reads the `0` in `0 4 * * 0` as Monday). The account's scope key is its user id, so its job ids predate the table. The table, not the Redis markers (written for diagnostics only), decides whether a scope is scheduled; `copilot/dream/registry.py` keeps it in step:

- **Registration.** Ingest registers a scope when its group's queue is created: the group's first write in a process, and again whenever the queue comes back after retiring (60 s idle), with at most one registration in flight per group. A hire or raise registers the expert in the background. Registration never reactivates a PAUSED or WIPED scope, and writes nothing while both of the owner's dream flags are off, so a scope that was never registered or paused has no row.
- **Lifecycle.** Archive and a schedule pause (including a budget breach) pause the expert's scope and remove its crons; the resume route and a revive resume a PAUSED scope (or register one that has no row). A lifecycle change the deadline cuts off is not retried: the cron gate still keeps an archived or paused expert's crons from running, and a later lifecycle change or the backfill settles the row (an ordinary registration does not look at the expert's lifecycle). `mark_wiped` is a seam for the wipe work: nothing calls it yet and it does not erase the graph. No registry operation takes a scope out of WIPED: registration and resume skip it, and pause or archive leave it WIPED (a pause only moves an ACTIVE or PAUSED row, checked in the same statement), so the wipe work decides what brings one back.
- **Timezone.** A timezone change re-registers every scope of the user in the new zone. If that fails, the next registration picks the new zone up (a returning queue, a lifecycle change, the backfill), not necessarily the next write while the group's worker is alive.
- **Deadlines.** Every registry call to the DatabaseManager, the scheduler, the flag backend or the user lookup runs under a 10 s deadline (`copilot/dream/deadline.py`), the Redis markers under their own 5 s one, all built on `asyncio.timeout` so a deadline nested in another still holds on Python 3.11. On timeout the call's cancellation is requested; a timeout does not prove the call had no effect, since the server may already have acted on it. The API hooks bound the whole change, so the memory part of a request ends at that deadline (plus any time the event loop itself is blocked), and a cron whose registry gate cannot answer in time skips that run. Background registration (ingest, hire, raise) and the backfill pay the deadline per call, so during an outage the backfill's run time grows with the number of scopes.

Existing deployments fill the table with `poetry run memory-schedule-backfill` (`--dry-run` only counts; `--force` re-registers every cron, e.g. after jobs were lost from the scheduler). It schedules every account with a `user_*` FalkorDB graph and a user row, resumes or registers every live hired expert, pauses paused and archived ones, and prints a JSON report that lists each failed scope with its cron and reason; it exits 1 if anything failed. Re-running is safe; it registers nothing twice, but pauses the paused experts again (a state write and a remove call each). It needs Postgres, Redis, FalkorDB, the scheduler service and the flag backend, so run it where the backend services' environment is set; crons are only registered for owners with `dream-pass-enabled` or `graphiti-communities-enabled` on.

**Deploy order.** Apply the migration and deploy the DatabaseManager and scheduler services first, then the writers (REST API, copilot-executor), then run the backfill: newer writers get a 404 from an older scheduler's scope RPCs, and a newer scheduler's cron gate skips every memory run, the account's included, while an older DatabaseManager cannot answer the registry. Expert crons pass `expert_id`, which an older scheduler's cron bodies do not accept, so they are not rollback-compatible; account crons are.

## Query Cookbook

Run everything from `autogpt_platform/backend` and use `poetry run ...`.

Get the `group_id` for a user (use `MemoryScope.for_expert(user_id, expert_id)`
for one of their experts; the snippets below take either):

```bash
poetry run python - <<'PY'
from backend.copilot.graphiti.scope import MemoryScope
print(MemoryScope.for_user("883cc9da-fe37-4863-839b-acba022bf3ef").group_id)
PY
```

Inspect graph counts:

```bash
poetry run python - <<'PY'
import asyncio
from backend.copilot.graphiti.falkordb_driver import open_driver
from backend.copilot.graphiti.scope import MemoryScope

USER_ID = "883cc9da-fe37-4863-839b-acba022bf3ef"
SCOPE = MemoryScope.for_user(USER_ID)

QUERIES = {
    "entities": "MATCH (n:Entity) RETURN count(n) AS count",
    "episodes": "MATCH (n:Episodic) RETURN count(n) AS count",
    "communities": "MATCH (n:Community) RETURN count(n) AS count",
    "relates_to_edges": "MATCH ()-[e:RELATES_TO]->() RETURN count(e) AS count",
}

async def run():
    driver = open_driver(SCOPE)
    try:
        for name, query in QUERIES.items():
            records, _, _ = await driver.execute_query(query)
            print(name, records[0]["count"])
    finally:
        await driver.close()

asyncio.run(run())
PY
```

List entities or relation-name counts:

```bash
poetry run python - <<'PY'
import asyncio
from backend.copilot.graphiti.falkordb_driver import open_driver
from backend.copilot.graphiti.scope import MemoryScope

USER_ID = "883cc9da-fe37-4863-839b-acba022bf3ef"
SCOPE = MemoryScope.for_user(USER_ID)

async def run():
    driver = open_driver(SCOPE)
    try:
        records, _, _ = await driver.execute_query(
            "MATCH (n:Entity) RETURN n.name AS name, n.summary AS summary ORDER BY n.name"
        )
        print("## entities")
        for row in records:
            print(row)

        records, _, _ = await driver.execute_query(
            """
            MATCH ()-[e:RELATES_TO]->()
            RETURN e.name AS relation, count(e) AS count
            ORDER BY count DESC, relation
            """
        )
        print("\\n## relation_counts")
        for row in records:
            print(row)
    finally:
        await driver.close()

asyncio.run(run())
PY
```

Inspect facts around one node:

```bash
poetry run python - <<'PY'
import asyncio
from backend.copilot.graphiti.falkordb_driver import open_driver
from backend.copilot.graphiti.scope import MemoryScope

USER_ID = "883cc9da-fe37-4863-839b-acba022bf3ef"
SCOPE = MemoryScope.for_user(USER_ID)
TARGET = "sarah"

async def run():
    driver = open_driver(SCOPE)
    try:
        records, _, _ = await driver.execute_query(
            """
            MATCH (a)-[e:RELATES_TO]->(b)
            WHERE (exists(a.name) AND toLower(a.name) = $target)
               OR (exists(b.name) AND toLower(b.name) = $target)
            RETURN a.name AS source, e.name AS relation, e.fact AS fact, b.name AS target
            ORDER BY e.created_at
            """,
            target=TARGET,
        )
        for row in records:
            print(row)
    finally:
        await driver.close()

asyncio.run(run())
PY
```

Inspect all chat messages for a user:

```bash
poetry run python - <<'PY'
import asyncio
from prisma import Prisma

USER_ID = "883cc9da-fe37-4863-839b-acba022bf3ef"

async def run():
    db = Prisma()
    await db.connect()
    try:
        rows = await db.query_raw(
            '''
            select cm."sessionId" as session_id,
                   cm.sequence,
                   cm.role,
                   left(cm.content, 260) as content,
                   cm."createdAt" as created_at
            from "ChatMessage" cm
            join "ChatSession" cs on cs.id = cm."sessionId"
            where cs."userId" = $1
            order by cm."createdAt", cm.sequence
            ''',
            USER_ID,
        )
        for row in rows:
            print(row)
    finally:
        await db.disconnect()

asyncio.run(run())
PY
```

Notes:

- `RELATES_TO` edges hold semantic facts. Inspect `e.name` and `e.fact`.
- `MENTIONS` edges are provenance from episodes to extracted nodes.
- Prefer directed queries `->` when checking for duplicates; undirected matches double-count mirrored edges.
