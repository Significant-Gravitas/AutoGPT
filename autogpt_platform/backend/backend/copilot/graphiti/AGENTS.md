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
- Be careful about memory pollution from assistant/tool phrasing; extraction quality matters as much as ingestion success.
- Keep warm-context and tool-driven recall resilient: failures should degrade gracefully rather than break chat execution.

## Query Cookbook

Run everything from `autogpt_platform/backend` and use `poetry run ...`.

Get the `group_id` for a user:

```bash
poetry run python - <<'PY'
from backend.copilot.graphiti.client import derive_group_id
print(derive_group_id("883cc9da-fe37-4863-839b-acba022bf3ef"))
PY
```

Inspect graph counts:

```bash
poetry run python - <<'PY'
import asyncio
from backend.copilot.graphiti.client import derive_group_id
from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver

USER_ID = "883cc9da-fe37-4863-839b-acba022bf3ef"
GROUP_ID = derive_group_id(USER_ID)

QUERIES = {
    "entities": "MATCH (n:Entity) RETURN count(n) AS count",
    "episodes": "MATCH (n:Episodic) RETURN count(n) AS count",
    "communities": "MATCH (n:Community) RETURN count(n) AS count",
    "relates_to_edges": "MATCH ()-[e:RELATES_TO]->() RETURN count(e) AS count",
}

async def run():
    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=GROUP_ID,
    )
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
from backend.copilot.graphiti.client import derive_group_id
from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver

USER_ID = "883cc9da-fe37-4863-839b-acba022bf3ef"
GROUP_ID = derive_group_id(USER_ID)

async def run():
    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=GROUP_ID,
    )
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
from backend.copilot.graphiti.client import derive_group_id
from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver

USER_ID = "883cc9da-fe37-4863-839b-acba022bf3ef"
GROUP_ID = derive_group_id(USER_ID)
TARGET = "sarah"

async def run():
    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=GROUP_ID,
    )
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
