# Memory Eval Plan — AgentProbe-style scaffolding

Plan for the test/eval infrastructure that validates P-1 (PR #13094) and lays the foundation for P0.6's benchmark harness. Driven by findings in `dreaming-agentprobe-evals.md`.

---

## What we have already

**On PR #13094 (P-1 in-tree):**
- `tools/graphiti_forget_test.py` — mock-driver tests for `_retract_edges`, `_soft_delete_edges`, `mark_edges_superseded`, `invalidate_entity_direct_neighbors`. They pin the Cypher string and call args; they do **not** execute against a live FalkorDB.
- `graphiti/communities_test.py` — mocked `rebuild_communities_for_user`; verifies DETACH DELETE precedes `build_communities`, verifies failure path returns error dict.
- `graphiti/client_test.py` — xfail regression for hyphenated group_id (Graphiti #1483).

These are unit tests. They tell us our Python code calls the right Cypher; they do **not** tell us our Cypher produces the right graph state under a real FalkorDB.

## What's missing

Two distinct test layers, neither in the PR yet:

**Layer A — Graph-correctness integration tests.** Pytest, live FalkorDB + Graphiti, validates P-1 behavior end-to-end. Direct test of "did we ship the right thing."

**Layer B — Conversational memory eval.** AgentProbe-vendored YAML scenarios + Python runner. Validates that memory behavior survives across sessions. Enabler for P0.6.

Both should ship. Layer A is the most direct validation of P-1 and unblocks the rest of the dream-chain PRs from regressing the audit fixes. Layer B is foundation work for P0 — won't validate P0 until P0 lands, but the scenarios it ships can already probe P-1 indirectly (the `mem-negative-forget-on-request` scenario at `multi-session-memory.yaml:1710` *is* a demotion-correctness probe).

---

## Layer A — Graph-correctness integration tests

**Goal:** for each P-1 behavior, write a pytest test that boots a real FalkorDB (docker), exercises the helper, and asserts the resulting graph state via raw Cypher.

**Fixture infrastructure** (new):

`backend/copilot/graphiti/integration_test_fixtures.py`:
- `falkordb_container` — module-scoped pytest fixture that runs FalkorDB in docker (or expects an already-running compose stack via `FALKORDB_INTEGRATION_HOST` env var).
- `clean_graph` — function-scoped fixture: creates a unique per-test `group_id`, returns `(client, driver, group_id)`, tears down by dropping the database after.
- `seeded_graph` — variant that pre-ingests a small canonical graph (3 entities, 2 edges) so tests can build on a known shape.

**Test files**:

| File | Tests | Validates |
|---|---|---|
| `graphiti/types_integration_test.py` | After `add_episode` with text containing "Alice works on Atlas," verify the resulting `:RELATES_TO` edge has `status='active'`, `source_kind` non-null, `scope` populated | P-1.1 wire-in actually puts properties on edges |
| `tools/graphiti_forget_integration_test.py` | `_retract_edges` followed by `MATCH (...)-[e:RELATES_TO]-(...) RETURN e.expired_at, e.invalid_at` — assert `expired_at != null` AND `invalid_at IS NULL`. Inverse for `_soft_delete_edges` | P-1.3 Snodgrass split |
| `tools/graphiti_forget_integration_test.py` (cont.) | After `mark_edges_superseded(reason="stale")`, verify edge has `status='superseded'` AND `expiration_reason='stale'` AND `expired_at != null` | P-1.3 audit trail |
| `tools/graphiti_forget_integration_test.py` (cont.) | **Hard test of single-hop discipline**: build A→B→C→D, call `invalidate_entity_direct_neighbors(entity_uuid=B)`, assert (A,B) and (B,C) are superseded, **assert (C,D) is untouched** | P-1.3 single-hop |
| `graphiti/communities_integration_test.py` | Seed ~10 entities + edges, call `rebuild_communities_for_user`, verify at least one `:Community` node exists with `group_id` matching | P-1.7 happy path |
| `graphiti/communities_integration_test.py` (cont.) | Run rebuild twice in succession, verify the second run cleanly replaces the first (no orphan `:Community` nodes from the prior run) | P-1.7 DETACH DELETE guard |
| `graphiti/migrations/backfill_edge_props_integration_test.py` | Pre-populate edges WITHOUT `status` (by writing raw without custom types), run `backfill_one_user`, verify all edges now have `status='active'` | P-1.2 idempotency and correctness |

**Discoverability:** mark these `@pytest.mark.integration` and `@pytest.mark.requires_falkordb`. CI runs them with the docker stack up; local devs run them only when they have the stack running.

**Skips:** if `FALKORDB_INTEGRATION_HOST` env var is unset and we cannot detect a running FalkorDB, skip the whole module with a clear `pytest.skip("integration env not available")` so unit-test runs stay fast.

**Effort estimate:** ~3 engineer-days. Mostly fixture infrastructure (1 day); per-behavior assertions are short.

**Ships:** as a follow-up PR on top of #13094, against `dev`, branch `feat/copilot-graphiti-audit-tests`.

---

## Layer B — Vendored AgentProbe memory scenarios

**Goal:** vendor `multi-session-memory.yaml` + the six memory rubrics from `AgentProbe/data/rubric.yaml:483-958`, build a Python runner that drives them against our chat backend, and emit metrics shaped for P0.6's harness.

**Vendoring policy** (per `dreaming-agentprobe-evals.md`):
- AgentProbe has **no LICENSE file** at the repo root. Both repos sit under `Significant-Gravitas`. Before vendoring, confirm with ownership that intra-org copy is fine. Document the source commit in the file header so we can pull fixes upstream.
- Copy as **data**, not code. The YAML packs and rubrics are 80% of the value and have no language coupling. The TypeScript runner we reimplement in Python.

**New tree:**

```
backend/copilot/dream/eval/                  # already referenced in p0-spec.md
├── __init__.py
├── runner.py                                # CLI: `poetry run dream-eval`
├── judge.py                                 # LLM judge with structured output + cache key
├── personas.py                              # simulator that drafts user turns
├── adapters.py                              # SUT adapter for our chat endpoint
├── metrics.py                               # scoring + aggregation
├── retrieval.py                             # NDCG, MRR, precision@k, recall@k (gap AgentProbe doesn't fill)
├── data/
│   ├── multi-session-memory.yaml            # vendored from AgentProbe a61df07
│   ├── memory-rubric.yaml                   # vendored (the 6 memory rubrics)
│   └── retrieval-golden.yaml                # new: golden query → expected memory uuid set
├── fixtures/
│   └── seeded_graphs/                       # new: pre-cooked graph dumps for retrieval evals
└── runner_test.py
```

**Runner contract** (mirrors AgentProbe's `runScenario` at a high level, simplified):
1. Read scenario YAML.
2. For each `session` in `scenarios`:
   - If `reset == fresh_agent`, mint a fresh `ChatSession` for a synthetic user; AgentProbe's "cold boot" maps to "new session_id, same user_id" in our system.
   - For each persona turn, simulator drafts a message, runner POSTs to `/sessions/{id}/stream`, collects the SSE stream into a transcript.
3. Judge the full multi-session transcript against the rubric.
4. Emit a `RunResult` JSON with per-rubric-dimension scores + named-failure-mode + cost + latency.

**Which scenarios first** (priority order from the 30):
1. `mem-negative-forget-on-request` (`multi-session-memory.yaml:1710`) — directly probes demotion correctness. **Validates P-1.3 indirectly.** Seed a budget figure in S1, user says "forget that" in S2, ask in S3 — `forget_acknowledged_but_leaked` is the explicit failure mode.
2. `mem-temporal-stale-team-member` (line 688) — tests stale-fact demotion. Validates P0.3a once P0 ships.
3. `mem-temporal-deprecated-procedure` (line 741) — same lineage as the gpt-image-2 example. Validates P0.3a.
4. `mem-retention-basic-identity` (line 48) — floor test. If this fails, the rest is meaningless.
5. `mem-retention-incidental-facts` (line 90) — slightly harder retention.
6. `mem-compositional-board-prep` (line 1305) — two-fact compositional reasoning. **The test that distinguishes graph memory from flat stores** per the YAML annotation. Validates our Graphiti choice end-to-end.

The remaining 24 land as a batch when P0 stabilizes.

**Retrieval metrics gap** (AgentProbe has none — explicit finding from the survey): we must build precision@k / recall@k / MRR / NDCG@k ourselves. Live in `eval/retrieval.py`. Golden query format:

```yaml
- query: "What is Sarah's email?"
  expected_memory_uuids:  # the dream pass / memory_search must return at least one of these in the top k
    - 11111111-…
    - 22222222-…
  k_threshold: 5
```

Effort: ~5 engineer-days. The judge + simulator are the heaviest pieces; the metrics module is half a day.

**Ships:** as a separate PR on top of #13094, branch `feat/copilot-dream-eval-scaffolding`. Independently mergeable from Layer A.

---

## Recommended order

1. **First, Layer A (graph-correctness pytest)** — direct validation of #13094, ~3 days, no external dependencies beyond docker. Lands as `feat/copilot-graphiti-audit-tests`.
2. **Confirm license/IP clearance** for vendoring AgentProbe YAML — quick async check with ownership.
3. **Then Layer B (eval scaffolding)** — ~5 days, depends on the cleared YAML vendoring. Lands as `feat/copilot-dream-eval-scaffolding`. Initially runs only the 6 priority scenarios; the rest of the 30 ride a follow-up.
4. **`mem-negative-forget-on-request` becomes a smoke test for P-1** once Layer B is live — runs in CI on every PR touching `copilot/graphiti/` or `copilot/tools/graphiti_*`.

## Open questions

1. **Where does the integration FalkorDB live in CI?** The existing docker-compose stack runs it; the `poetry run test` invocation per `backend/AGENTS.md` already spins up postgres+prisma via docker. Need to confirm FalkorDB is in that stack already or needs adding. If already there, fixture work shrinks meaningfully.
2. **Who reviews the rubric ports?** AgentProbe rubrics target their conversational SUT; our SUT is different. Reviewer should check rubric language doesn't bake in AgentProbe-specific terminology.
3. **Cost budget for nightly eval runs?** Each conversational scenario is ~10 LLM calls (simulator + adapter) plus 1 judge call. 30 scenarios nightly is ~$30/day at Opus pricing. Likely fine but flag for FinOps.
4. **Does Layer A block P0 work?** No — P0 dream-pass code can be written against the typed edges from #13094 regardless of whether Layer A lands. But Layer A *should* land before P0 ships to production so we have a regression net.
