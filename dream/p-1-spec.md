wh# P-1 — Graphiti Audit Fixes + Community Enablement

**Base of the dream-system PR chain.** Against `dev`. Sits next to PR [#12993](https://github.com/Significant-Gravitas/AutoGPT/pull/12993) (local transport) — neither depends on the other.

Source: `dream/dreaming-graphiti.md` audit §6 + user direction to enable communities.

---

## Why this is P-1 (not P0)

The Graphiti audit surfaced two HIGH-severity items that touch every existing `add_episode` call site, not just future dream-pass writes. Folding them into P0 would conflate "ship dreaming" with "fix our Graphiti integration." Better to land the cleanup first as its own reviewable PR, then P0 builds on a clean foundation.

The two high items:

1. **Edge metadata is stranded in `:Episodic.content` JSON.** `status`, `confidence`, `scope`, `source_kind`, `provenance` are written into the envelope JSON we attach as the episode body, but they never make it onto the LLM-extracted `:RELATES_TO` edges. Search can't filter on them. Ratification can't flip them efficiently. Every read path has to parse JSON to recover the field.
2. **`_soft_delete_edges` violates Snodgrass bi-temporal semantics.** It sets BOTH `expired_at` (transaction time — system retracted) AND `invalid_at` (valid time — world changed). Those are different concepts. User-initiated forget is a system retraction; only `expired_at` should fire. Conflating them means we can't tell "we stopped tracking it" from "it stopped being true."

Plus the user-directed change: **enable community detection with scheduled rebuilds**, behind a feature flag.

## What changes

### P-1.1 — Custom entity + edge types

**New file:** `backend/copilot/graphiti/types.py`

```python
from datetime import datetime
from pydantic import BaseModel, Field
from backend.copilot.graphiti.memory_model import MemoryStatus, SourceKind


class MemoryFact(BaseModel):
    """Custom Graphiti edge type for our domain.

    Attached to :RELATES_TO edges so MemoryEnvelope metadata survives
    the LLM extraction step and lives on the durable graph edge, not
    only in the :Episodic.content JSON blob (which can't be filtered
    in Cypher without parsing).
    """
    status: MemoryStatus = Field(default=MemoryStatus.active)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source_kind: SourceKind = Field(default=SourceKind.user_asserted)
    scope: str = Field(default="real:global")
    provenance: str | None = Field(default=None)
    web_verified_at: datetime | None = Field(default=None)
    ratified_at: datetime | None = Field(default=None)
    expiration_reason: str | None = Field(default=None)


# Entity types — start narrow; expand only when we see extraction quality issues
class Person(BaseModel):
    role: str | None = None
    email: str | None = None


class Organization(BaseModel):
    industry: str | None = None


class Project(BaseModel):
    status: str | None = None


class Concept(BaseModel):
    pass


class Preference(BaseModel):
    pass


class Rule(BaseModel):
    pass


ENTITY_TYPES = {
    "Person": Person,
    "Organization": Organization,
    "Project": Project,
    "Concept": Concept,
    "Preference": Preference,
    "Rule": Rule,
}

EDGE_TYPES = {"MemoryFact": MemoryFact}

# Allow MemoryFact between any of our entity types
EDGE_TYPE_MAP = {
    (src, tgt): ["MemoryFact"]
    for src in ENTITY_TYPES
    for tgt in ENTITY_TYPES
}
```

**Wire-in:** every `client.add_episode(...)` call site passes `entity_types=ENTITY_TYPES, edge_types=EDGE_TYPES, edge_type_map=EDGE_TYPE_MAP`. Sites today:

- `backend/copilot/graphiti/ingest.py` — `_add_episode_to_graphiti` (the one we control)

After this, search filters can do `WHERE e.status = 'active'` natively.

### P-1.2 — Backfill migration

**New file:** `backend/copilot/graphiti/migrations/2026_05_dream_edge_props.py`

For every existing `:RELATES_TO` edge that lacks the new properties, set defaults:

```cypher
MATCH ()-[e:RELATES_TO]-()
WHERE e.status IS NULL
SET e.status = 'active'
```

Run via `poetry run python -m backend.copilot.graphiti.migrations.2026_05_dream_edge_props` against each user's FalkorDB database. Idempotent (re-running is a no-op).

Must run **before** any P-1.1 reads start using the new properties. Document this in the migration script's docstring and in the PR description.

### P-1.3 — Split `_soft_delete_edges`

**Edit:** `backend/copilot/tools/graphiti_forget.py`

Today's `_soft_delete_edges` sets `invalid_at = datetime(), expired_at = datetime()`. Split:

```python
async def _retract_edges(driver, uuids: list[str]) -> int:
    """System retraction: only expired_at. Use for user forget, dream demotion,
    entity invalidation. Per Snodgrass bi-temporal model."""

async def _soft_delete_edges(driver, uuids: list[str]) -> int:
    """Reserved for the contradiction detector: both expired_at AND invalid_at.
    Means the world changed (invalid_at) AND we recorded it (expired_at)."""

async def mark_edges_superseded(
    driver,
    uuids: list[str],
    reason: str,
    new_status: Literal["superseded", "contradicted"] = "superseded",
) -> int:
    """Retract + set audit-trail status property."""

async def invalidate_entity_direct_neighbors(
    driver,
    group_id: str,
    entity_uuid: str,
    reason: str,
) -> list[str]:
    """Single-hop demotion of all edges attached to an entity. P0.3b prep."""
```

Existing callers of `_soft_delete_edges` (user-initiated forget) **switch to `_retract_edges`**. Update call sites in:

- `MemoryForgetConfirmTool` (in `graphiti_forget.py` itself)

The two new helpers (`mark_edges_superseded`, `invalidate_entity_direct_neighbors`) are not called from anywhere yet — they're API surface for P0 to call into.

### P-1.4 — Cross-encoder for warm context

**Edit:** `backend/copilot/graphiti/context.py`

Replace `client.search(query=..., num_results=..., group_ids=...)` with the explicit configurable variant:

```python
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_CROSS_ENCODER

results = await client.search_(
    query=query,
    config=NODE_HYBRID_SEARCH_CROSS_ENCODER,
    group_ids=group_ids,
)
```

~10–15% precision lift on the warm-context call at the cost of one extra LLM judge call per session. Worth it for the *one* most-impactful retrieval per session.

Explicit `MemorySearchTool` (tools/graphiti_search.py) keeps the default `search()` for now — that's an interactive cost lever, separately tunable.

### P-1.5 — Richer provenance grain

**Edit:** `backend/copilot/tools/graphiti_store.py`

Change provenance from `session.session_id` to `f"session:{session.session_id}#msg:{message_sequence}"`. The sequence number is available from the current message context (`message.sequence`).

Trivial change, ~3 lines.

### P-1.6 — Hyphenated group_id regression test

**New test:** `backend/copilot/graphiti/client_test.py` (extension)

```python
@pytest.mark.asyncio
async def test_derive_group_id_handles_hyphenated_uuid():
    """Guards against Graphiti issue #1483: FalkorDriver fulltext fails on
    hyphens in group_id. Our derive_group_id allows hyphens (UUIDs contain them);
    this test ingests + searches with a hyphenated UUID to verify end-to-end."""
    uuid_with_hyphens = "a1b2c3d4-e5f6-7890-1234-567890abcdef"
    group_id = derive_group_id(uuid_with_hyphens)
    assert "-" in group_id  # we intentionally preserve hyphens

    client = await get_graphiti_client(group_id)
    # ingest a trivial episode
    await client.add_episode(
        name="probe", episode_body="Alice works on Project Atlas.",
        source=EpisodeType.text, group_id=group_id,
        reference_time=datetime.now(timezone.utc),
    )
    # search must return results without raising on the hyphen
    results = await client.search(query="Alice", group_ids=[group_id], num_results=5)
    assert len(results) > 0
```

If this fails, the fix is to sanitize hyphens → underscores in `derive_group_id`. Until upstream fixes #1483, the test is the canary.

### P-1.7 — Community detection + scheduled rebuilds

**New file:** `backend/copilot/graphiti/communities.py`

```python
async def rebuild_communities_for_user(user_id: str) -> dict:
    """Destroy and rebuild :Community nodes for a single user's graph.

    Per the multi-episode research, upstream build_communities() calls
    remove_communities() first — but older versions did not. We add a
    defensive explicit DETACH DELETE before rebuild to be safe across
    version bumps.
    """
    group_id = derive_group_id(user_id)
    client = await get_graphiti_client(group_id)

    # Defensive: clean up any orphan :Community nodes regardless of version
    cleanup = """
    MATCH (c:Community {group_id: $group_id})
    DETACH DELETE c
    """
    await client.driver.execute_query(cleanup, group_id=group_id)

    # Rebuild via Graphiti
    summary = await client.build_communities(group_ids=[group_id])
    return {"user_id": user_id, "communities_built": summary}
```

**Scheduler hook:** new `@expose` methods in `backend/executor/scheduler.py`:

```python
@expose
def add_community_rebuild_schedule(
    self, user_id: str, user_timezone: str = "UTC"
) -> JobInfo:
    """Weekly community rebuild at user-local 4am Sunday.

    Off-peak to avoid the Leiden cost spike during active hours.
    Staggered from the (future) 3am dream pass so they don't contend
    on the same FalkorDB writes."""
    job = self.scheduler.add_job(
        execute_community_rebuild_sync,
        kwargs={"user_id": user_id},
        trigger=CronTrigger.from_crontab("0 4 * * 0", timezone=user_timezone),
        id=f"community_rebuild_{user_id}",
        max_instances=1,
        replace_existing=True,
        jobstore=Jobstores.EXECUTION.value,
    )
    return JobInfo.from_job(job)

@expose
def execute_community_rebuild_pass(self, user_id: str) -> dict:
    return run_async(rebuild_communities_for_user(user_id))

@expose
def delete_community_rebuild_schedule(self, user_id: str) -> bool:
    return self.scheduler.remove_job(f"community_rebuild_{user_id}")

def execute_community_rebuild_sync(user_id: str) -> dict:
    return run_async(rebuild_communities_for_user(user_id))
```

**Feature flag:** `GRAPHITI_COMMUNITIES_ENABLED` (LaunchDarkly). Off by default. Scheduler skips the cron registration when the flag is off for the user. Allow per-user opt-in for internal canary.

**Operational note:** `build_communities()` is `O(|V| · log|V|)` plus one LLM call per community for summarization. For a heavy user with thousands of entities this can take minutes. The `max_instances=1` per user + the 4am Sunday cron keep contention bounded.

## Files touched

| File | Change | Audit ref |
|---|---|---|
| `backend/copilot/graphiti/types.py` | **new** | §6.4 |
| `backend/copilot/graphiti/ingest.py` | pass entity/edge types to `add_episode` | §6.4 |
| `backend/copilot/graphiti/migrations/__init__.py` | **new** (package init) | §6.4 |
| `backend/copilot/graphiti/migrations/2026_05_dream_edge_props.py` | **new** | §6.4 |
| `backend/copilot/tools/graphiti_forget.py` | split soft delete; add helpers | §6.13 |
| `backend/copilot/tools/graphiti_forget_test.py` | update + add tests for new helpers | §6.13 |
| `backend/copilot/graphiti/context.py` | `search_()` + cross-encoder | §6.7 |
| `backend/copilot/tools/graphiti_store.py` | richer provenance grain | §6.12 |
| `backend/copilot/graphiti/client_test.py` | hyphenated UUID regression test | §6.1 |
| `backend/copilot/graphiti/communities.py` | **new** | new |
| `backend/copilot/graphiti/communities_test.py` | **new** | new |
| `backend/executor/scheduler.py` | community rebuild `@expose` methods | new |
| `backend/executor/scheduler_test.py` | community rebuild schedule test | new |

## Order of work in the PR (commits)

1. `feat(backend/copilot): add MemoryFact edge type + custom entity types`
2. `feat(backend/copilot): wire entity_types/edge_types into add_episode calls`
3. `feat(backend/copilot): add Cypher backfill migration for existing edges`
4. `refactor(backend/copilot): split _soft_delete_edges into _retract_edges + retain original for contradiction detector`
5. `feat(backend/copilot): add mark_edges_superseded + invalidate_entity_direct_neighbors helpers`
6. `feat(backend/copilot): use search_() + cross-encoder for warm context`
7. `feat(backend/copilot): include message sequence in MemoryStore provenance`
8. `test(backend/copilot): regression test for hyphenated group_id (Graphiti #1483)`
9. `feat(backend/copilot): enable community detection with scheduled rebuilds`

Each commit independently green (`poetry run test`).

## PR description outline

**Title:** `feat(backend/copilot): graphiti integration audit fixes + community detection`

**Why:** Audit (`dream/dreaming-graphiti.md`) found two high-severity issues with our Graphiti integration that block downstream work on the dream pass: stranded edge metadata in `:Episodic.content` JSON blobs that search can't filter on, and a soft-delete that conflates transaction-time retraction with valid-time world-change. Also enables Graphiti's community detection (off by default, opt-in via feature flag) so future memory consolidation has a structural clustering signal.

**What:** See P-1.1–P-1.7 above.

**How:** Custom Graphiti entity + edge types (`MemoryFact` carries the envelope metadata); Cypher backfill for existing edges; split `_soft_delete_edges` into `_retract_edges` (system) and the original (world); switch warm-context retrieval to the cross-encoder recipe; richer provenance; regression test for hyphenated group_ids; scheduler job for weekly community rebuilds behind LD flag `GRAPHITI_COMMUNITIES_ENABLED`.

**Risk:** Medium. Edge schema additions are additive and backwards-compatible via the migration. The soft-delete split changes the *meaning* of an existing call but keeps the function signature; carefully review every caller in the PR. Cross-encoder change adds one LLM call per session — measurable cost bump, validated against current warm-context latency budget.

**Out of scope:**
- Dream pass itself (P0).
- The cross-encoder for explicit memory search (kept on default `search()` — separate cost lever).
- A retrieval ablation eval for cross-encoder lift (P0.6 will cover it).
- Removing the `:Episodic.content` JSON mirror of the same metadata (kept for backward compat; remove in a v2 PR once all readers consume edge properties).
