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
- Recall goes through `recall.py` (what may be read back) and `recall_render.py` (how it is written out). Read facts with `search_facts` and episodes with `recent_episodes`; ingestion hands graphiti `previous_episode_uuids`. Code with its own Cypher uses the predicates there: `live_fact_predicate` for facts (the settings and admin fact lists, the dream's demotion writers), `forgotten_facts_clause` with `recallable_episode_predicate` for episodes (the dream gather and its session filter, the forget's redaction).
  - A fact is live while `expired_at` and `forgotten_at` are unset and `status` is `active`, `tentative` or unset. It is forgotten once a forget has stamped `forgotten_at`. Only a forget writes that marker (`recall_forget.py`, and the legacy-forget backfill for older forgets), so no dream demotion or other status change can make a forgotten fact look remembered; graphiti rewriting a forgotten edge during ingestion can drop it, and the ingestion repair below puts it back. Older forgets are recognised too: `retracted`, an `expiration_reason` of `user_signal`, or the pre-policy shape (`expired_at` set with no `invalid_at` and no reason). `migrations/backfill_legacy_forgets.py` restamps that last shape, after which its clause can go. An episode is recallable while it has no `redacted_at` and none of its `entity_edges` is forgotten.
  - What a forget guarantees: the fact and every episode it came from are hidden from warm context, `memory_search`, `memory_forget_search`, the settings fact list and counts, the dream's input (its episodes and facts, and every chat session a hidden episode came from, left out whole) and ingestion's extraction context (the earlier episodes graphiti shows its prompts). The sentence also leaves what graphiti's own prompts read. The edge's `fact` and `name` read `[forgotten]` (`recall.FORGOTTEN_FACT`) and the originals move to `fact_redacted` / `name_redacted`; no edge type has that name, so graphiti never runs its attribute prompt, which lists an edge's stored properties, on a forgotten edge. The `summary` of both endpoint entities, and of every community either belongs to, is blanked, not rewritten: graphiti grows an entity summary back from the next episode that mentions it, and the weekly rebuild restores the communities.
  - What it does not: admin audit surfaces (status `any`, the graph view) keep showing retracted facts, with the text from `fact_redacted`, and redacted episodes, on purpose; a tombstone shows only its `hard_deleted_at`. Entity attributes, community names, the summary of an entity the fact only mentioned and the fact's embedding vector are left as they are. A hard forget keeps, redacted, the text of an episode another edge still cites, and a tombstone keeps its name (for a stored memory, the model's short title). A forget that lands while graphiti is writing the same edge in an `add_episode` can be overwritten by that write (graphiti saves the edge it read earlier); the ingestion repair below only guards forgets made before the episode started.
  - Every forget goes through `recall_forget.retract`. Soft stamps `forgotten_at`, sets `status='retracted'` and `expiration_reason`, keeps an earlier `expired_at`, leaves `invalid_at` alone, then scrubs the sentence and summaries and stamps `redacted_at` on every episode citing the fact (`recall_hide.py`). Hard does all of that first, then (`recall_orphans.py`) turns each episode no other edge of any status cites into a tombstone (`content` emptied, `hard_deleted_at` and `redacted_at` set, its name, source description and envelope `provenance` kept, so the dream still leaves its chat session out), and last, in one query per edge, deletes the edge, drops the tombstones' mentions and deletes each entity left with no fact and no mention. Every step before the delete is idempotent and the delete finds the edge again, so repeating a forget whose clean-up failed finishes it. A step that fails is reported per edge as `cleanup_error`; recall hides the text regardless. A tombstone is never recalled, counted on the settings page or used as extraction context.
  - Ingestion keeps a forget (`recall_ingest.py`). graphiti resolves a new fact against every edge between the same entities, forgotten ones included: its model can call the new fact a duplicate of a forgotten edge (graphiti then appends the new episode and rewrites the edge's attributes, dropping the forget's fields) or say it contradicts one (graphiti stamps `invalid_at`), and either way the new episode cites the forgotten edge and is hidden with it. So `ingest._add_episode` snapshots every forgotten edge before `add_episode` and afterwards puts back each one graphiti changed, points the new episode away from it, and gives a fact stated again a new live edge: graphiti's own edge extraction re-read from the episode, `MemoryFact` defaults, `valid_at` from the episode, and the episode as its only source (the worker then stamps a dream write's envelope metadata on it like any new edge). A failed snapshot or repair is logged and never fails the write.
  - The dream keeps its own Cypher. Its gather and session filter (`dream/fetch.py`, `dream/hidden_sessions.py`) build on those predicates. Its writers only write over live facts: supersession (`mark_edges_superseded`) and single-hop neighbour invalidation (`invalidate_entity_direct_neighbors`) match `live_fact_predicate` unless a caller passes `expected_status`, and ratification (`dream/ratification.py`) passes `expected_status='tentative'` and promotes only a still-tentative, unexpired edge, so none of them can overwrite a forget.
  - A live fact whose valid time has ended (a past `invalid_at`, never expired) is still recalled, as history: it renders as `fact (valid: X — Y)`. Only a retired fact gets a retirement label, and "present" appears only while no end is known.

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
