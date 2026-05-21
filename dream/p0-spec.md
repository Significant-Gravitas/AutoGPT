# P0 — Day-One Build Spec

Concrete engineering plan for the seven P0 items in `TODO.md`. Pulls in details already covered by `dreaming-batch.md`, `dreaming-memory.md`, `dreaming-scheduler.md`, `dreaming-chat.md`, `dreaming-spec.md`, and `dreaming-anthropic.md`; fills the gaps not yet specified.

Goal: an engineer can pick this up Monday and start writing code without re-reading the research bundle.

> **READY GATE — DO NOT BUILD UNTIL CLEARED**
>
> Graphiti research has landed (`dream/dreaming-graphiti.md`) and its findings are folded into the spec below. One blocker remaining:
>
> 1. ~~`dream/dreaming-graphiti.md`~~ **DONE.** Findings folded into §2 (custom entity/edge types so `status`/`confidence`/`scope`/`source_kind` live on edges, not stranded in `:Episodic.content`), §4 (use `expired_at` not `invalid_at` for system retractions; new `_retract_edges` helper), §5 (status flips also write to edge property), §6 (web-verified facts get `web_verified_at` edge property). Two high-severity audit items now P0: stranded edge metadata + Snodgrass-correct soft-delete split.
>
> 2. **PR [#12993](https://github.com/Significant-Gravitas/AutoGPT/pull/12993)** — local transport. Still gating; see §13.
>
> Once #12993 lands, re-read this spec end-to-end before opening the first feature PR.

---

## 0. Architecture overview

Three execution paths in the platform, all reachable from the chat executor and the scheduler:

```
┌──────────┐    ┌──────────────┐    ┌─────────────────┐    ┌───────────────────┐
│  Chat    │───▶│ enqueue_     │───▶│ copilot_executor│───▶│ baseline or SDK   │
│  routes  │    │ copilot_turn │    │ (sync, SSE)     │    │  → OpenRouter     │
└──────────┘    └──────┬───────┘    └─────────────────┘    └───────────────────┘
                       │
                       │ if mode="batch" or "dream"
                       ▼
                ┌──────────────┐    ┌──────────────────────┐    ┌──────────────────┐
                │ MessageBatch │───▶│ copilot_batch_       │───▶│ anthropic.com    │
                │ Item row     │    │ executor (poll loop) │    │ /v1/messages/    │
                └──────────────┘    └──────────────────────┘    │ batches          │
                                                                └──────────────────┘
```

Dream pass = a scheduled job that mints a `mode="dream"` `CoPilotExecutionEntry`, gets routed to batch, and applies the results back via `enqueue_episode()` + raw Cypher demotions.

**Five new directories, all under `autogpt_platform/backend/backend/copilot/`:**
- `batch/` — Anthropic batch client, accumulator, poller, results parser
- `dream/` — dream pass body, prompts, output schemas
- `eval/` — memory benchmark harness (P0.6)
- `webcheck/` — web fact-check tool (P0.5)
- `ratification/` — ratification scheduler job (P0.4)

Existing files that get edited (not rewritten):
- `copilot/config.py` — add `CopilotMode = Literal["fast", "extended_thinking", "dream", "daydream", "batch"]` plus new `ChatConfig` fields
- `copilot/executor/utils.py` — extend `CoPilotExecutionEntry` minimally; route `mode in {dream, batch}` to batch path
- `copilot/graphiti/ingest.py` — add `enqueue_dream_episode()` wrapper; pass `entity_types` and `edge_types` to every `add_episode` call (see §2 custom types)
- `copilot/graphiti/types.py` — **new file**. Defines `MemoryFact` edge type with `status`, `confidence`, `source_kind`, `scope`, `provenance`, `web_verified_at`, `ratified_at`, `expiration_reason`, plus entity types (`Person`, `Organization`, `Project`, `Concept`, `Preference`, `Rule`). Per Graphiti audit item #1.
- `copilot/tools/graphiti_forget.py` — split `_soft_delete_edges` into `_retract_edges` (sets only `expired_at` — system retraction, per Snodgrass) and keep `_soft_delete_edges` (sets both for contradiction detector). Add `mark_edges_superseded()` and `invalidate_entity_direct_neighbors()`. Per Graphiti audit item #2.
- `copilot/tools/permissions.py` — add `DREAM_PERMISSIONS` preset
- `executor/scheduler.py` — add `add_dream_pass_schedule`, `execute_dream_pass`, `run_ratification_pass`
- `data/model.py` — extend `ChatSessionMetadata` (JSON only, no migration)
- `schema.prisma` — new `MessageBatch`, `MessageBatchItem` models
- `monitoring/instrumentation.py` — new Prometheus instruments
- **One-time Cypher backfill script** at `backend/copilot/graphiti/migrations/2026_05_dream_edge_props.py` — adds default `status='active'` to existing `:RELATES_TO` edges. Must run before P0.2 ships.

---

## 1. P0.1 — Anthropic direct batch processing

### Files to create
- `backend/copilot/batch/__init__.py`
- `backend/copilot/batch/client.py` — thin `anthropic.AsyncAnthropic().messages.batches` wrapper with retries, rate-limit handling, structured logging. ~150 LOC.
- `backend/copilot/batch/accumulator.py` — consumes `copilot_batch_pending` RabbitMQ queue, flushes on size (N=100) or time (5 min default). Redis-backed pending buffer for restart safety. ~200 LOC.
- `backend/copilot/batch/poller.py` — restartable poll loop. State in Redis (`batch:active:{batch_id}` → `{status, custom_id_map, submitted_at, expires_at}`). Backoff: 30s → 5min, capped at 5min. ~250 LOC.
- `backend/copilot/batch/results.py` — streams JSONL from `/results`, routes each result by `custom_id` back to the per-item handler. ~150 LOC.
- `backend/copilot/batch/service_test.py`, `batch_e2e_test.py` — colocated.

### New executable
- `backend/copilot/batch_executor/__main__.py` — entry point for the new service. Mirrors `backend/copilot/executor/__main__.py`.

### Docker compose
Append to `autogpt_platform/docker-compose.platform.yml` after the `copilot_executor` block:

```yaml
copilot_batch_executor:
  <<: *backend-defaults
  command: ["python", "-u", "-m", "backend.copilot.batch_executor"]
  container_name: copilot_batch_executor
  depends_on:
    db:
      condition: service_healthy
    redis:
      condition: service_healthy
    rabbitmq:
      condition: service_healthy
```

### Prisma schema additions
```prisma
model MessageBatch {
  id                String              @id @default(uuid())
  anthropicBatchId  String?             @unique
  status            String              // "pending" | "submitted" | "in_progress" | "ended" | "expired" | "canceled"
  submittedAt       DateTime?
  endedAt           DateTime?
  expiresAt         DateTime?
  resultsUrl        String?
  itemCount         Int
  errorCount        Int                 @default(0)
  expiredCount      Int                 @default(0)
  totalCostUsd      Decimal?            @db.Decimal(10, 4)
  createdAt         DateTime            @default(now())
  updatedAt         DateTime            @updatedAt
  items             MessageBatchItem[]

  @@index([status])
  @@index([createdAt])
  @@map("message_batches")
}

model MessageBatchItem {
  id                String              @id @default(uuid())
  batchId           String
  batch             MessageBatch        @relation(fields: [batchId], references: [id], onDelete: Cascade)
  customId          String              // unique within batch: "{user_id}:{turn_id}"
  userId            String?
  sessionId         String?
  turnId            String
  mode              String              // "dream" | "daydream" | "batch_chat"
  requestPayload    Json
  resultStatus      String?             // "succeeded" | "errored" | "canceled" | "expired"
  resultPayload     Json?
  inputTokens       Int?
  outputTokens      Int?
  cacheReadTokens   Int?
  cacheWriteTokens  Int?
  createdAt         DateTime            @default(now())
  completedAt       DateTime?

  @@unique([batchId, customId])
  @@index([userId])
  @@index([turnId])
  @@map("message_batch_items")
}
```

Migration: `prisma migrate dev --name add_message_batches`.

### RabbitMQ
New exchange + queue:
- `copilot_batch_pending` (DIRECT, durable) — accumulator consumes
- Routing key: `copilot.batch.pending`

Both declared in `backend/copilot/batch/client.py` constants alongside `COPILOT_EXECUTION_EXCHANGE`.

### Anthropic SDK
Add to `pyproject.toml`:
```toml
anthropic = "^0.40.0"  # check current; we want batch API support
```
API key resolution: `ANTHROPIC_API_KEY` env → `config.anthropic_api_key`. Separate from `OPENROUTER_API_KEY`.

### Settings additions (`backend/util/settings.py`)
```python
class ChatConfig(BaseSettings):
    # ... existing fields ...
    anthropic_api_key: str = Field(default="")
    batch_processing_enabled: bool = Field(default=False)
    batch_flush_max_items: int = Field(default=100)
    batch_flush_max_seconds: int = Field(default=300)
    batch_poll_max_concurrent: int = Field(default=5)
    batch_poll_initial_interval_seconds: int = Field(default=30)
    batch_poll_max_interval_seconds: int = Field(default=300)
```

### LaunchDarkly flag
`BATCH_PROCESSING_DREAMS` (bool, default off). Gates whether the scheduler routes dream passes through batch.

---

## 2. P0.2 — Memory recombination (the dream pass body)

### Files to create
- `backend/copilot/dream/__init__.py`
- `backend/copilot/dream/pass.py` — orchestrator: fetch context, build prompt, submit to batch or sync, parse output, apply operations. ~300 LOC.
- `backend/copilot/dream/prompts.py` — three Langfuse-fallback prompt builders (see below).
- `backend/copilot/dream/schemas.py` — Pydantic models for the structured output the three phases emit.
- `backend/copilot/dream/apply.py` — applies `DreamOutput` to Graphiti + Postgres (writes tentative memories, demotes by uuid, updates `ChatSessionMetadata.last_dream_at`).
- `backend/copilot/dream/fetch.py` — input gathering: `list_episodes_in_window`, `list_active_facts`, `fetch_recent_sessions`, all bounded.
- `backend/copilot/dream/locks.py` — Redis SETNX advisory lock for `dream_lock:{group_id}` with 30-min TTL.
- `backend/copilot/dream/pass_test.py` — unit tests with all external calls mocked.

### Three-phase pipeline

Phase 1 — **Consolidation** (Sonnet, temp 0.2, structured output):
- Input: ≤50 recent episodes + ≤500 active facts, scope-grouped
- Output: list of `ConsolidatedFact` (clusters of related facts merged into a single canonical statement, with provenance pointing to source episode uuids)

Phase 2 — **Recombination** (Opus, temp 0.9, extended thinking, freeform):
- Input: phase 1 output + user's active goals (from `MemoryKind.plan` + open agents)
- Output: list of `ProposedFinding` (novel connections, weak-link discoveries, tentative claims) — `status=tentative` until ratified

Phase 3 — **Sanitize + Gate** (Sonnet, temp 0.0, deterministic structured output):
- Input: phase 1 + phase 2 outputs combined
- Output: final `DreamOperations` — list of writes (tentative envelopes), demotions (by uuid), and entity invalidations (uuid + reason); enforces guardrails: ≤10 demotions per pass, scope match enforced, first-person-as-user regex rejection, entity allowlist verification

### Bounded inputs (concrete numbers)
| Input | Cap | Rationale |
|---|---|---|
| Recent sessions | 10, max 8 KB each | matches Anthropic *Dreams* ≤100 sessions × smaller-scale per-user shard |
| Recent episodes | 50 | spec.md §2 |
| Active facts (per scope) | 500 | bounds prompt size |
| Touch window | 14 days | recency bias |
| Phase 1 output | ≤4k tokens | bounds phase 2 input |
| Phase 2 output | ≤8k tokens | bounds phase 3 input |
| Demotions per pass | ≤10 | runaway-demotion guardrail |

Total prompt size target: ~120 KB. Well below batch's 256 MB limit.

### Schemas (sketch)

```python
class ConsolidatedFact(BaseModel):
    content: str
    scope: str
    confidence: float
    source_episode_uuids: list[str]

class ProposedFinding(BaseModel):
    content: str
    scope: str
    memory_kind: MemoryKind  # finding | rule | preference | plan
    confidence: float
    rationale: str  # why the dream pass proposes this
    source_episode_uuids: list[str]
    source_fact_uuids: list[str]

class DreamDemotion(BaseModel):
    edge_uuid: str
    reason: str  # "stale_fact" | "contradicted_by:{uuid}" | "entity_invalidated:{uuid}" | "user_signal"
    new_status: Literal["superseded", "contradicted"]

class EntityInvalidation(BaseModel):
    entity_uuid: str
    reason: str  # short user-facing reason

class DreamOperations(BaseModel):
    writes: list[ConsolidatedFact]
    proposals: list[ProposedFinding]
    demotions: list[DreamDemotion]
    entity_invalidations: list[EntityInvalidation]
    summary_for_user: str  # short narrative for the dream-kind session
```

### Output application (`apply.py`)
- For each `ConsolidatedFact` and `ProposedFinding`: build a `MemoryEnvelope` with `source_kind=assistant_derived`, `provenance="dream:{pass_id}:{ts}:session:{src_session}#msg:{src_seq}"` (richer than just session id — per Graphiti audit item #6), `status=tentative` for proposals, `status=active` for consolidated facts. Call `enqueue_dream_episode()`.
- For each `DreamDemotion`: call `mark_edges_superseded(driver, [uuid])` — which sets `expired_at = datetime()`, `status = 'superseded'`, `expiration_reason = <reason>` on the edge. **Does NOT set `invalid_at`** — that's reserved for world-changes per Snodgrass bi-temporal semantics (Graphiti audit §6.13).
- For each `EntityInvalidation`: call `invalidate_entity_direct_neighbors(driver, entity_uuid, reason)` — same `expired_at` + `status` + `expiration_reason` pattern, single-hop only.
- Write a `ChatSession` with `metadata.kind="dream"`, `metadata.dream_pass_id`, append a single `ChatMessage(role="assistant", content=summary_for_user)`.

### Write path discipline (Graphiti audit §7.1)
- **Use `add_episode()`, never `add_triplet()`.** `add_triplet` skips temporal invalidation and has open bug #1001 against FalkorDB.
- **Pass `update_communities=False` explicitly.** We do not run communities in P0; passing False guards against a future Graphiti default flip.
- **`reference_time` = `valid_at` of the most recent episode in the input window**, not `datetime.now()`. The dream consolidates *historical* facts; the system should think they were learned at the time of their evidence.
- **`name` prefix = `dream_{pass_id}_{phase}_{counter}`** so dream-derived episodes are auditable and selectively removable.
- **Pass `entity_types` and `edge_types` from `copilot/graphiti/types.py` on every call** so `status`/`confidence`/`scope`/`source_kind`/`provenance` survive on `:RELATES_TO` edges and search can filter on them natively. This eliminates the "stranded metadata in `:Episodic.content` JSON blob" anti-pattern (Graphiti audit §6.4, item #1 high).

---

## 3. P0.3a — Stale-fact deprecation

### Detection
Heuristic flags surfaced to phase 3 sanitizer (not blocking — sanitizer can override based on context):
- Age > 90 days
- Contains version numbers (regex `v?\d+\.\d+|^\d{4}$|GPT-\d|claude-\d|gpt-image-\d`)
- Contains "best" / "current" / "latest" / "as of" / "now"
- Names a specific model, API version, price, leader of a company

These heuristics live in `backend/copilot/dream/staleness.py` as `score_staleness(envelope: MemoryEnvelope) -> float`. Returns 0.0–1.0; ≥0.6 surfaces the fact to phase 3 with a "consider for demotion" flag.

### Confirmation
Phase 3 sanitizer prompt receives both the high-staleness candidates and the consolidated facts from phase 1; the LLM judges whether each candidate is actually contradicted by a phase 1 fact or by general knowledge, and emits a `DreamDemotion` only when it is.

### Edge case
Stale-but-still-correct facts (e.g., "user's birthday is March 5") must not be demoted. Test fixture: a dataset of 50 hand-curated facts split between "stale and wrong" (e.g., "GPT-4 is the best model in 2024") and "old but stable" (birthdays, anniversaries, foundational preferences). Phase 3 must demote the first set and preserve the second. This is the first item in the P0.6 benchmark suite.

---

## 4. P0.3b — Scoped cascading expiry

### Two new helpers in `backend/copilot/tools/graphiti_forget.py`

```python
async def _retract_edges(driver, uuids: list[str]) -> int:
    """
    System-retraction soft delete. Sets ONLY expired_at (transaction-time).
    Per Snodgrass bi-temporal semantics, this means 'we retracted the record',
    NOT 'the world changed' (which would set invalid_at).

    Use for: dream-driven demotion, user-initiated forget, entity invalidation.
    Do NOT use for: contradiction-detector edge invalidation (use _soft_delete_edges).
    """
    query = """
    MATCH ()-[r]-()
    WHERE r.uuid IN $uuids AND type(r) IN ['MENTIONS','RELATES_TO','HAS_MEMBER']
    SET r.expired_at = datetime()
    RETURN count(r) AS n
    """
    return (await driver.execute_query(query, uuids=uuids))[0]["n"]


async def mark_edges_superseded(
    driver,
    uuids: list[str],
    reason: str,
    new_status: Literal["superseded", "contradicted"] = "superseded",
) -> int:
    """Retract edges AND set the custom audit-trail status property."""
    query = """
    MATCH ()-[r:RELATES_TO]-()
    WHERE r.uuid IN $uuids
    SET r.expired_at = datetime(),
        r.status = $new_status,
        r.expiration_reason = $reason
    RETURN count(r) AS n
    """
    return (await driver.execute_query(
        query, uuids=uuids, new_status=new_status, reason=reason,
    ))[0]["n"]


async def invalidate_entity_direct_neighbors(
    driver,
    group_id: str,
    entity_uuid: str,
    reason: str,
) -> list[str]:
    """
    Demote all RELATES_TO edges directly attached to an entity.
    Single-hop only — does NOT propagate to neighbors-of-neighbors.
    """
    query = """
    MATCH (e:Entity {uuid: $entity_uuid, group_id: $group_id})
    MATCH (e)-[r:RELATES_TO]-(other)
    SET r.status = 'superseded',
        r.expired_at = datetime(),
        r.expiration_reason = $reason
    RETURN r.uuid AS edge_uuid
    """
    result = await driver.execute_query(
        query,
        entity_uuid=entity_uuid,
        group_id=group_id,
        reason=reason,
    )
    return [r["edge_uuid"] for r in result]
```

**Critical**:
- `MATCH (e)-[r:RELATES_TO]-(other)` is **single hop**. Do not write `[r:RELATES_TO*1..N]` — that's the runaway-demotion bug.
- These helpers set `expired_at` (transaction-time, system retraction) and **never** `invalid_at` (valid-time, world change). Per Snodgrass [1999] and the Graphiti audit §6.13, conflating the two is a semantic bug.
- Existing `_soft_delete_edges` (which sets both) is retained for the contradiction-detector path only.

### Exposed to the dream pass as a phase 3 action
When phase 3 emits an `EntityInvalidation`, `apply.py` calls this helper. No user-facing tool; it's strictly internal to the dream pass.

### Tests
- Set up a 3-hop graph: A → B → C → D, all `RELATES_TO`.
- Invalidate B.
- Assert edges A↔B and B↔C are superseded.
- Assert edge C↔D is **not** touched.
- This is a hard correctness gate.

---

## 5. P0.4 — Ratification loop

### New scheduler job in `backend/executor/scheduler.py`

```python
self.scheduler.add_job(
    run_ratification_pass,
    trigger=IntervalTrigger(hours=6),
    id="ratification_pass",
    max_instances=1,
    replace_existing=True,
    jobstore=Jobstores.EXECUTION.value,
)
```

### Body in `backend/copilot/ratification/pass.py`

For each user with at least one `tentative` memory:
1. Fetch all `tentative` memories with `provenance` starting with `dream:` (created > 24h ago, < 30d ago).
2. For each, check whether the memory's fact text or any of its `source_episode_uuids` appears in any `ChatMessage.content` from the last 7 days OR in any warm-context retrieval call in the last 7 days (instrumented separately).
3. **If hit and age > 24h**: flip `status=active` **on both the `:RELATES_TO` edge AND the `:Episodic.content` JSON** (Graphiti audit §7.3 — edge property is what `search()` can filter, content blob is for backward-compatible JSON readers). Set `ratified_at = datetime()` on the edge. Sub-10ms Cypher per [11].
4. **If age > 30 days and no hits**: call `mark_edges_superseded(driver, [edge_uuid], reason="unratified")` — flips edge status and sets `expired_at`. Also mutate `:Episodic.content` in parallel.

```cypher
// Ratify (status active):
MATCH ()-[e:RELATES_TO {uuid: $edge_uuid}]->()
SET e.status = 'active', e.ratified_at = datetime()
```

### Warm-context hit tracking
Add a Redis hash `mem:hits:{user_id}` with key=`memory_uuid` and value=timestamp. Written from `fetch_warm_context()` in `graphiti/context.py` whenever a memory is returned to a chat turn. TTL: 14 days.

### Metric
Ratification rate = `count(status flipped to active in last 30d) / count(tentative memories created in last 30d)`. Surface in `monitoring/instrumentation.py` as `dream_ratification_rate`.

---

## 6. P0.5 — Web-search-backed fact verification

### New tool in `backend/copilot/tools/web_fact_check.py`

```python
class WebFactCheckTool(BaseTool):
    name = "web_fact_check"
    description = (
        "Verify a time-sensitive factual claim against current web results. "
        "Returns: 'confirmed' | 'contradicted' | 'unverifiable' + supporting URLs. "
        "ONLY usable inside the dream pass; never exposed to user chat."
    )
    parameters = {
        "claim": "string — the exact claim to verify",
        "context": "string — surrounding context that defines the scope of the claim",
    }
    requires_auth = False

    async def _execute(self, user_id, session, claim, context):
        # 1. Use existing web_fetch tool / search API to get top 5 results
        # 2. LLM judge: given (claim, context, top results), is the claim confirmed/contradicted/unverifiable
        # 3. Return structured WebFactCheckResult with reasoning + URLs
```

### Restricted via `CopilotPermissions`
In `backend/copilot/tools/permissions.py` define a `DREAM_PERMISSIONS` preset that allows only:
- `memory_search`, `memory_store`, `memory_forget_search`, `memory_forget_confirm`
- `web_fact_check`

The dream pass passes this preset on `enqueue_copilot_turn(permissions=DREAM_PERMISSIONS)`.

### Scope guard (re-stated)
The web fact-check tool can only **demote** memories. Any *new* fact suggested by the web is written as `tentative` and rides the ratification loop (P0.4). Phase 3 sanitizer enforces this.

### Write path for web-verified facts (Graphiti audit §7.4)
- **New facts confirmed by the web** are written via `add_episode()` with `source_kind=tool_observed` (matches the existing `SourceKind` enum at `memory_model.py:14-17`; no new source type needed). The episode body preserves the URL(s), verification timestamp, and the judge's reasoning — provenance-complete per Lewis 2020.
- **`web_verified_at` is set as an edge property** on the resulting `:RELATES_TO` edges. Search filters that want "only web-verified facts" can `WHERE e.web_verified_at IS NOT NULL`. No JSON parsing needed.
- **Demotion case**: the contradicted edge gets `status='contradicted'`, `expiration_reason='web_contradicted:{url}'` via `mark_edges_superseded()`. The fresh fact arrives as its own new episode.

### Search backend
Reuse existing `web_fetch` infrastructure where possible. If we need search-engine queries (not single-URL fetches), add a thin wrapper around an existing provider (Brave Search API, Tavily, etc.) — defer choice to product; the tool interface is provider-agnostic.

---

## 7. P0.6 — Memory benchmark harness

### Why this is P0 not P1
Without P0.6, we cannot tell whether any of P0.1–P0.5 are net positive. **Snapshot baseline metrics before P0.1–P0.5 ship.**

### Files to create
- `backend/copilot/eval/__init__.py`
- `backend/copilot/eval/runner.py` — CLI runner: `poetry run dream-eval --suite all --out results.json`
- `backend/copilot/eval/datasets/` — JSON files, one per eval. Versioned; never edit in place (write `v2`, mark old).
- `backend/copilot/eval/judges.py` — LLM-judge prompt templates with structured output.
- `backend/copilot/eval/metrics.py` — scoring functions (precision, recall, MRR, NDCG, ratification rate).
- `backend/copilot/eval/suites/`
  - `retrieval.py` — warm-context relevance (golden query + expected memory uuid set)
  - `demotion.py` — demotion precision (synthetic memory set + which should be demoted)
  - `ratification.py` — ratification rate (offline replay of 30d of memories + chat)
  - `web_fact_check.py` — web fact-check P/R against curated ground-truth examples
  - `cost.py` — token + dollar accounting per dream pass on a fixed input
  - `latency.py` — p50/p95 timing across 100 synthetic dream passes

### CLI
```
poetry run dream-eval --suite all --baseline baseline.json --out results.json
```
- `--baseline` compares against a previously saved run; emits a diff with per-metric % change.
- `--suite {name}` to run a single suite.

### Output schema
```json
{
  "run_id": "uuid",
  "ts": "2026-05-12T03:00:00Z",
  "git_sha": "abc123",
  "graphiti_version": "0.28.2",
  "transport": "openrouter",
  "suites": {
    "retrieval": {"ndcg_at_5": 0.62, "mrr": 0.71, "n_queries": 50},
    "demotion": {"precision": 0.88, "recall": 0.55, "n_examples": 50},
    "ratification": {"rate_7d": 0.34, "rate_30d": 0.62, "n_memories": 500},
    "web_fact_check": {"precision": 0.91, "recall": 0.78, "n_claims": 30},
    "cost": {"usd_per_pass_mean": 0.31, "usd_per_pass_p95": 0.52},
    "latency": {"p50_seconds": 124, "p95_seconds": 380}
  }
}
```

**`graphiti_version` is recorded on every benchmark run** (Graphiti audit §7.5). Upstream changes its extraction prompts between versions; a regression in warm-context relevance may be ours OR theirs. Pin the version in `pyproject.toml`, log it here, review on every bump.

### AgentProbe potential reuse
A separate research agent is surveying `/Users/ntindle/code/agpt/AgentProbe` for memory evals we can lift, extend, or learn from. Findings will land in `dream/dreaming-agentprobe-evals.md`. Wait for that before locking in `eval/runner.py` design — if AgentProbe has a usable runner we want to inherit it rather than reinvent.

### Golden dataset curation
For each suite, curate 30–100 examples by hand from real user transcripts (sanitize PII first, per project memory). Treat datasets as code — review in PRs, version, regenerate annually.

---

## 8. Scheduler wiring (`backend/executor/scheduler.py`)

```python
@expose
def add_dream_pass_schedule(self, user_id: str, user_timezone: str) -> JobInfo:
    job = self.scheduler.add_job(
        execute_dream_pass_sync,
        kwargs={"user_id": user_id},
        trigger=CronTrigger.from_crontab("0 3 * * *", timezone=user_timezone),
        id=f"dream_pass_{user_id}",
        max_instances=1,
        replace_existing=True,
        jobstore=Jobstores.EXECUTION.value,
    )
    return JobInfo.from_job(job)

@expose
def delete_dream_pass_schedule(self, user_id: str) -> bool:
    return self.scheduler.remove_job(f"dream_pass_{user_id}")

@expose
def execute_dream_pass(self, user_id: str) -> dict:
    """On-demand trigger; bypasses cron. Same body as the scheduled job."""
    return run_async(_execute_dream_pass(user_id))

def execute_dream_pass_sync(user_id: str) -> dict:
    return run_async(_execute_dream_pass(user_id))
```

`_execute_dream_pass()` does:
1. Acquire Redis lock (SETNX `dream:inflight:{user_id}` EX 1800). Bail if held.
2. Mint a `dream` session and `turn_id`.
3. Call `enqueue_copilot_turn(mode="dream", permissions=DREAM_PERMISSIONS, ...)` — gets routed to batch by the executor's mode check.
4. **Return immediately**. The batch poller will pick up the result hours later and call `dream.apply.apply_operations()`.

Note: this means the scheduler thread is released in milliseconds. The actual work happens in the batch executor. Confirms the architecture from `dreaming-scheduler.md` §6 ("scheduler is the wrong place to run the dream body").

---

## 9. Tests

### Unit tests (per file)
- `batch/client_test.py` — mock `anthropic.AsyncAnthropic`; verify request shape, error mapping.
- `batch/accumulator_test.py` — verify flush triggers (size + time).
- `batch/poller_test.py` — verify backoff schedule, Redis state hydration on restart.
- `batch/results_test.py` — verify JSONL parsing + per-`custom_id` routing.
- `dream/pass_test.py` — mock all external calls; verify three-phase orchestration.
- `dream/apply_test.py` — verify each `DreamOperations` variant produces the right Graphiti calls.
- `dream/staleness_test.py` — heuristic-only unit tests.
- `ratification/pass_test.py` — mocked Redis hit tracker.
- `webcheck/tool_test.py` — mock web_fetch + judge.
- `tools/graphiti_forget_test.py` — add `test_invalidate_entity_direct_neighbors` with the 3-hop graph fixture from §4.

### Integration tests
- `dream/integration_test.py` (using `SpinTestServer`) — schedules a dream, mocks Anthropic batch API, verifies end-to-end flow lands memories in FalkorDB.

### Eval-as-test
Run `dream-eval --suite all` in CI on every PR touching `dream/` or `eval/`. Regress on any metric > 10% triggers a failure. Baseline updated only via explicit `--update-baseline` flag in a separate PR.

---

## 10. Rollout

### Feature flags (LaunchDarkly)
- `BATCH_PROCESSING_DREAMS` — master switch. Off by default.
- `DREAM_PASS_ENABLED_USERS` — list of user IDs for canary. Internal team first.
- `DREAM_PASS_WEB_FACT_CHECK` — gates P0.5 separately (search API cost concern).
- `DREAM_PASS_INVALIDATE_ENTITY` — gates P0.3b. **Off for first 2 weeks** — direct-neighbor demotion is the highest-blast-radius action; verify on canary before broad rollout.

### Sequence
1. Week 1–2: Prisma migration + batch client + Anthropic SDK. Internal e2e test against a sample batch.
2. Week 3–4: Accumulator + poller. Standing batch infra runs even with no dream consumers.
3. Week 4–6: Dream pass body + apply + tools. Three-phase pipeline tested offline against curated transcripts.
4. Week 6–7: Scheduler wiring + Redis lock. Internal canary (5 users) gets nightly dreams.
5. Week 7–8: Eval harness + benchmark baseline. P0.6 lands alongside the canary widening.
6. Week 8+: Expand canary to 50 → 500 → 5k. Watch metrics.

### Kill switches
Every numbered subsystem has its own LD flag. If any metric regresses, flip that flag off and stop the bleeding without disabling the rest.

---

## 11. Open items for the engineer

1. **Anthropic SDK pinning** — verify the version currently in `pyproject.toml` (if any) supports batches; if not, version-bump and confirm no SDK breakage.
2. **`copilot_executor` routing logic** — exact change in `backend/copilot/executor/manager.py` to route `mode in {dream, batch}` to the batch accumulator instead of running through the standard turn path. Single function: `route_entry(entry: CoPilotExecutionEntry) -> Literal["sync", "batch"]`.
3. **Search backend for P0.5** — pick one of Brave/Tavily/Serper/Google. Cost + ToS comparison; defer to product.
4. **AgentProbe integration** — `dreaming-agentprobe-evals.md` landed. Decision: vendor the YAML packs (`data/multi-session-memory.yaml` + memory rubrics) under `backend/copilot/dream/eval/`, reimplement the runner in Python. Add a CI lane that drives the docker-composed backend with AgentProbe's `data/autogpt-endpoint.yaml`. License clearance from `Significant-Gravitas` ownership is the only blocker.
5. **Langfuse prompt seeds** — Phase 1/2/3 prompts go into Langfuse with `CoPilot Dream Phase {N}` names. Coordinate with the team owning Langfuse prompts.
6. **Cost dashboard** — Grafana panel for batch spend; coordinate with FinOps.
7. **Cypher backfill** — write `migrations/2026_05_dream_edge_props.py` that adds default `status='active'` to existing `:RELATES_TO` edges where the property is missing. Must run before §2 P0.2 writes anything with custom edge types. Per Graphiti audit §6.4.
8. **Custom types in upstream extraction** — pass `entity_types={...}` and `edge_types={"MemoryFact": MemoryFact}` on EVERY `add_episode` call site, not just dream-pass writes. Otherwise normal-chat episode ingestion silently keeps stranding metadata. Audit every caller of `add_episode` and `enqueue_episode` in the codebase before flipping over.
9. **`search_()` cross-encoder for warm context** (Graphiti audit §6.7) — replace default `search()` in `context.py` with `search_()` + `NODE_HYBRID_SEARCH_CROSS_ENCODER` recipe. ~10–15% precision lift at the cost of one extra LLM call per session. Not strictly P0, but ship in the same PR window — it's a no-touch win.
10. **Hyphenated group_id regression test** (Graphiti audit §6.1, issue #1483) — CI test that ingests + searches with a hyphenated UUID `group_id`. If broken, replace hyphens with underscores in `derive_group_id` until upstream fixes.

11. **Chat-stream `dream.operations` SSE event** — required for P6 (surface dreams in chat UI) and P9 (daydreaming). Today `DreamOperations` is only returned via the admin endpoint. When P6/P9 lands, dream + daydream orchestrators emit a structured `dream.operations` SSE event on the active session's stream (gated on `mode in {dream, daydream}`); frontend renders as a "dream/daydream summary" artifact. P0.6 eval does NOT need this — AgentProbe captures `DreamOperations` via the admin endpoint instead. Build P6 + this plumbing together. See `TODO.md` P9 note.

---

## 12. What this spec deliberately does NOT cover

- **Frontend changes** — surfacing dreams in the UI is P6 in the roadmap, not P0. The dream session is created in Postgres but no frontend work is required for P0 success.
- **User-facing trigger UI** — no "run dream now" button in P0; the scheduler is the only entry point.
- **Multi-episode summaries** — separate research pass (`dreaming-multi-episode.md`); not P0.
- **Procedure synthesis** — separate research pass (`dreaming-procedures.md`); not P0.
- **Cross-scope discovery** — P8, requires P2 dedup first.

The minimum viable dream pass goes to production with: scheduled overnight, no UI, web-fact-check on, entity-invalidation off for 2 weeks, eval harness running on every build.

---

## 13. Transport compatibility — local LLM support

PR [#12993](https://github.com/Significant-Gravitas/AutoGPT/pull/12993) adds a `local` transport alongside the existing `openrouter`, `subscription`, and `openai_compat` profiles. The dream pass is inference; it must work on all four. **The architectural rule: routing decisions branch on `config.transport.name`, never on a hard-coded provider check.**

### What each transport supports

| Transport | Batch (`/v1/messages/batches`) | SDK / extended thinking | Tool calls | Notes |
|---|---|---|---|---|
| `openrouter` | ✗ (OpenRouter doesn't expose batches) | ✓ | ✓ | Cloud via OpenRouter |
| `subscription` | ✗ (Claude Code subscription, no batch endpoint) | ✓ | ✓ | Cloud via Claude Agent SDK CLI |
| `openai_compat` | ✗ | ✗ | ✓ | Cloud, OpenAI-compatible proxy |
| `local` | ✗ | ✗ (gate at PR #12993 sets `supports_sdk=False`) | ✓ (baseline path only) | Ollama / vLLM / LocalAI / LM Studio / LiteLLM-proxy. Per PR #12993: vendor validator skipped, `local_num_ctx` forwarded as `options.num_ctx`, 1800 s request timeout, `CHAT_API_KEY` empty |

**Batch is only viable when we have a direct Anthropic API key** (i.e., not when `transport in {openrouter, subscription, openai_compat, local}`). For P0, this means batch is gated on a *separate* env var (`ANTHROPIC_API_KEY` distinct from `CHAT_API_KEY`) and only runs when present. **Anthropic key absent → all dreams fall back to sync baseline regardless of transport name.**

### Routing decision in the dream scheduler

```python
def resolve_dream_execution_path(config: ChatConfig) -> Literal["batch", "sync_baseline"]:
    if config.anthropic_api_key and config.batch_processing_enabled:
        return "batch"
    return "sync_baseline"
```

This lives in `backend/copilot/dream/routing.py`. Every dream pass calls it; the result determines whether the entry is routed to `copilot_batch_pending` or to the standard `copilot_executor` sync queue.

### What changes for sync_baseline dreams

When a dream pass runs sync through the baseline path:

1. **No `mode="extended_thinking"`.** PR #12993's `resolve_use_sdk_for_mode` already downgrades extended_thinking → fast under non-SDK transports with a logged WARNING. The dream pass uses `mode="fast"` for all three phases on local.
2. **Phase-collapse option for slow/weak backends.** New `ChatConfig.dream_single_phase_under_local: bool = True`. When set and `transport.name == "local"`, phases 1+2 are combined into a single prompt (still ending with phase 3 sanitizer). Trades quality for ~3x speedup on a CPU-only host. Defaults on for local; off for cloud.
3. **Per-transport model selection.** A new `dream_model_override` field on `ChatConfig` defaults to `None` (use whatever the transport's `fast_standard_model` resolves to). On local, the same model the user already runs handles dreams — no auto-pull. On openrouter / subscription, we default to Opus 4.7 for phase 2 and Sonnet 4.6 for phases 1+3 (per `dreaming-spec.md` §1).
4. **Timeout budget.** PR #12993's `local_request_timeout_s = 1800` is per-request; a three-phase dream pass on CPU can hit ~90 minutes. The scheduler's per-job `max_instances=1` plus the Redis SETNX lock (`dream:inflight:{user_id}` EX 7200) need a 2-hour TTL on local, not 30 min. New: `dream_lock_ttl_seconds`, defaults 1800 for cloud / 7200 for local.
5. **Web fact-check tool.** Still functional — it uses an external search API (HTTP) + an LLM judge (local-compatible). Slower but correct. The search API key is independent of the LLM key.
6. **Embeddings.** Graphiti's `OpenAIEmbedder` reads `CHAT_OPENAI_API_KEY` → `OPENAI_API_KEY` (per `dreaming-memory.md` §7). PR #12993 makes embedding model and dim env-overridable for the store path; we should mirror that for Graphiti — new `GRAPHITI_EMBEDDER_BASE_URL`, `GRAPHITI_EMBEDDER_MODEL`, `GRAPHITI_EMBEDDER_DIM` so local users can point at `nomic-embed-text` on Ollama (or whatever they prefer). Defaults to OpenAI for backward compat.

### What gets disabled or downgraded on local

- **P0.5 web fact-check is OPTIONAL on local.** New flag `dream_web_fact_check_enabled: bool = True`. Local users without a search-API key can flip it off; dream still runs, just without external verification. Stale-fact detection (P0.3a) still works on internal contradictions.
- **P0.6 eval harness must run on local.** This is the regression gate that catches "we shipped a change that broke local dreams." CI matrix: `transport=openrouter` baseline + `transport=local` (mocked Ollama) baseline. The local matrix accepts a wider quality bound — see `eval/runner.py` `--accept-quality-floor=local|cloud` flag.

### What this changes upstream in the spec

- §1 Architecture overview gets a fourth path: "sync baseline dream" via the existing `copilot_executor`, distinct from "batch dream" via `copilot_batch_executor`. The `enqueue_copilot_turn` call sets `mode="dream"`; the executor's mode dispatcher consults `resolve_dream_execution_path()` and routes accordingly.
- §8 Scheduler wiring — `_execute_dream_pass()` becomes transport-aware:

  ```python
  async def _execute_dream_pass(user_id: str) -> dict:
      ttl = config.dream_lock_ttl_seconds  # 1800 cloud / 7200 local
      async with dream_lock(group_id_for(user_id), ttl=ttl):
          path = resolve_dream_execution_path(config)
          session = await create_chat_session(user_id, dry_run=False)
          # ... mint metadata, build seed message
          await enqueue_copilot_turn(
              session_id=session.session_id,
              user_id=user_id,
              message=seed_message,
              turn_id=str(uuid4()),
              is_user_message=False,
              mode="dream",
              permissions=DREAM_PERMISSIONS,
              context={"dream_phase": "consolidate", "execution_path": path},
          )
  ```

- §10 Rollout — feature flags grow by one:
  - Existing: `BATCH_PROCESSING_DREAMS`, `DREAM_PASS_ENABLED_USERS`, `DREAM_PASS_WEB_FACT_CHECK`, `DREAM_PASS_INVALIDATE_ENTITY`
  - New: `DREAM_PASS_LOCAL_TRANSPORT` — opt-in for the local-transport dream path. Default off until we've validated end-to-end on at least one local-LLM contributor's stack.

### Why this is fair to open-source contributors

The PR #12993 contributors are the population that runs AutoPilot keyless on Ollama. Dreaming is "basically a form of inference," so if we ship it cloud-only we silently exclude every local install. Two acceptable outcomes for a local contributor:

1. They never see dreams (we gate `dream_pass_enabled: bool` per-user; off by default for local installs in v1). Acceptable but lazy.
2. They get a degraded dream pass: one-phase, slower, no web-fact-check unless they configure it, smaller `max_active_facts` cap to keep the prompt within `local_num_ctx`. **This is the goal.**

We ship (2). It will not be Opus-quality. It will not catch every stale fact. It will work, and it will be honest about its limits in the dream-summary message.

### What we explicitly DON'T do

- We do **not** try to make batch work on local. The Anthropic batch endpoint is Anthropic-API-only; no equivalent exists in OpenAI-compat. Local users get sync, period.
- We do **not** try to replicate extended-thinking on local. PR #12993's downgrade-to-fast is the correct UX. The dream pass's three phases compensate by giving the model multiple bites at the problem instead of one long bite.
- We do **not** require local users to install a search-API key. Web-fact-check is a *cloud-bias* feature; on local it's opt-in.
