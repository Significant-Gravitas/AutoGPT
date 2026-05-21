# Dream Roadmap

Ships in many parts. **P-1 is the foundation PR** (against `dev`) the rest of the chain depends on. P0 is the day-one commit; P1 is day-two; everything after is a prioritized queue.

---

## P-1 — Graphiti Audit Fixes + Community Enablement (foundation PR)

Direct outcome of `dreaming-graphiti.md` §6 audit. **This is the base of the dream-system PR chain** — all P0+ work merges on top. Sits next to PR [#12993](https://github.com/Significant-Gravitas/AutoGPT/pull/12993) (local transport), also against `dev`, neither depends on the other.

See `dream/p-1-spec.md` for the implementation spec.

| # | Item | Audit ref |
|---|---|---|
| P-1.1 | **`MemoryFact` edge type + entity types** — new `backend/copilot/graphiti/types.py`; pass `entity_types` + `edge_types` on every `add_episode` call site so `status`/`confidence`/`scope`/`source_kind`/`provenance` live on `:RELATES_TO` edges, not stranded in `:Episodic.content` JSON | High §6.4 / §6.15 #1 |
| P-1.2 | **Backfill migration** — Cypher script to set defaults on existing `:RELATES_TO` edges before custom-types reads start | §6.4 |
| P-1.3 | **Split `_soft_delete_edges`** — new `_retract_edges` (only `expired_at`; system retraction per Snodgrass) and keep `_soft_delete_edges` (both `expired_at`+`invalid_at`; reserved for contradiction detector). Plus `mark_edges_superseded` and `invalidate_entity_direct_neighbors` helpers | High §6.13 / §6.15 #2 |
| P-1.4 | **`search_()` + cross-encoder for warm context** — replace default `search()` in `context.py` with `search_()` + `NODE_HYBRID_SEARCH_CROSS_ENCODER` recipe | Medium §6.7 / §6.15 #4 |
| P-1.5 | **Richer provenance grain** — `session:{id}#msg:{sequence}` instead of bare `session_id` on `MemoryStoreTool` writes so ratification can find originating message | Low §6.12 / §6.15 #6 |
| P-1.6 | **Hyphenated group_id regression test** — guard against upstream issue #1483 | Medium §6.1 / §6.15 #3 |
| P-1.7 | **Enable community detection + scheduled rebuilds** — new `backend/copilot/graphiti/communities.py`, new scheduler job `rebuild_communities_pass(user_id)` running per-user weekly at user-local 4am Sunday, feature-flagged. Includes defensive pre-rebuild cleanup (`MATCH (c:Community) DETACH DELETE c`) for older Graphiti versions | New scope per user direction |

Communities were originally "leave off in P0, revisit if relevance plateaus" per the audit. User direction overrides: **enable now**, ship behind LD flag `GRAPHITI_COMMUNITIES_ENABLED` defaulting off, opt-in canary. Run during off-peak (per-user 4am Sunday) to avoid the Leiden cost spike during user activity.

**Build estimate:** ~2 engineer-weeks. Each P-1 item lands as its own commit on the same branch; PR opens when all green.

---

## P0 — Day-One Scope

| # | Item | Source doc |
|---|---|---|
| P0.1 | **Anthropic direct batch processing** — new `copilot_batch_executor` service, bypass OpenRouter for batch path | `dreaming-batch.md` §3, §5 |
| P0.2 | **Memory recombination** — phase-1 consolidation + phase-2 recombination over recent episodes; outputs tentative `MemoryEnvelope`s | `dreaming-spec.md` §1, §4 |
| P0.3a | **Stale-fact deprecation** — recognize fact-rot and demote (canonical example: "gpt-image-2 is the best image model" goes stale fast; dream pass detects via age + contradictory evidence + LLM judgment) | `dreaming-spec.md` §3 (Demotions) |
| P0.3b | **Scoped cascading expiry** — when an entity is invalidated ("this client is dead to us"), expire memories **directly attached** to that entity; **do not** propagate transitively to tangentially-related entities. Implemented as a graph-distance-bounded demotion: only edges where the target entity is source or target, not edges 2+ hops away | `dreaming-memory.md` §5; `dreaming-spec.md` §6 (runaway demotion mitigation) |
| P0.4 | **Ratification loop** — close the tentative→active flow. Track whether a `tentative` memory gets hit by warm-context retrieval or referenced by a user message in the next 7d; if yes flip `status=active`; if 30d pass with no hits, demote to `superseded` | `dreaming-spec.md` §5 |
| P0.5 | **Web-search-backed fact verification** — `web_fact_check` tool restricted to the dream pass; flag time-sensitive claims (model names, prices, API versions, leadership at companies), verify against current web results, demote on contradiction with `provenance="dream:web_verified:{ts}:{url}"` | extends P0.3a |
| P0.6 | **Memory benchmark harness** — without this we cannot tell whether P0.1–P0.5 are net positive. Standing eval suite covering: warm-context retrieval relevance (golden query set per user archetype, scored offline), demotion precision (sampled human/LLM-judge label of "should this have been demoted"), ratification rate (% of `tentative` memories used within 7d), web-fact-check precision/recall against curated ground-truth examples, end-to-end cost per dream pass, p50/p95 latency. Snapshot baseline before P0 ships so we can measure lift | new — not in original research; needed to gate every subsequent item |

**P0.3b implementation note** — Graphiti's natural traversal is transitive. We must explicitly clamp demotion to direct neighbors of the invalidated entity. Concretely: `MATCH (e:Entity {uuid: $dead_entity})-[r:RELATES_TO]-(other) SET r.status='superseded'` — single hop only, no `*1..N` expansion. Tangential edges stay active.

**P0.5 scope guard** — the `web_fact_check` tool can only **demote** memories on contradiction; any *new* facts the web suggests get written as `tentative` and flow through the ratification loop (P0.4). Avoids the dream pass becoming an unattended web-scraping agent.

**P0.6 staffing note** — benchmark harness can run partly in parallel with P0.1–P0.5 because much of it is golden-set curation and judge-prompt design rather than blocking on the dream pass itself.

---

## P1 — Day-Two Scope: Procedure synthesis + "Save as Skill"

**Promoted from P5.** Pattern detection in chat history → structured `ProcedureMemory` writes, plus a user-facing "Save as Skill" tool that lets users (and the dream pass) export curated procedures as first-class, reusable skills.

**Status**: deep research in flight — see `dream/dreaming-procedures.md` (5M-token budget research pass). Wait for that doc before scoping the build.

**Why day-two**: massive value-per-engineering-effort for power users; we already have `MemoryKind.procedure` in the envelope and nothing writes it; "skills" as a first-class platform concept is a natural extension of what `ProcedureMemory` was designed for.

**Dependency tension**: procedures normally benefit from dedup (P2). Workaround: the procedure pass does ad-hoc dedup of the patterns it identifies before persisting, rather than waiting for global dedup. The research doc will recommend a path.

---

## Post-P1 queue — ordered by recommended landing

Difficulty: 1 = trivial, 5 = research-grade. Days are rough engineer-days.

### P2 — Memory dedup / near-duplicate merge *(diff 3, ~1w)*
Memory bloat is the next operational pain after consolidation lands. Use `fact_embedding` cosine similarity > 0.92 to find candidates, LLM judge to confirm, merge with provenance union.
**Source**: `dreaming-memory.md` §8 Priority 4.
**Depends on**: P0.

### P3 — Self-model refresh *(diff 2, ~3d)*
Same batch infra, very high value, no new types. Regenerate `BusinessUnderstanding` (Postgres) from the last N episodes during the dream pass.
**Source**: implicit in `dreaming-chat.md` §4 (`<user_context>` injection); maps to Hobson's protoconsciousness / DMN self-simulation.
**Depends on**: P0.

### P4 — Scenario pre-warm (sleep-time compute) *(diff 3, ~1w)*
Predict the user's likely next 3–5 questions from recent activity, run them through batch overnight, cache the answers keyed by `(user_id, query_hash)`. Letta paper showed 5x test-time compute reduction.
**Source**: `dreaming-research.md` Part III (Sleep-time Compute, arXiv 2504.13171).
**Depends on**: P0.

### P5 — Goals-aware dreaming (motivational priming) *(diff 2, ~3d)*
Extract the user's active objectives (open agents, recent `MemoryKind.plan`, stated intents from the last N days) and bias the consolidation prompt to prioritize them. Small prompt change, large signal-to-noise win.
**Source**: `dreaming-research.md` Part II synthesis §5 (motivational priming).
**Depends on**: P0.

### P6 — Surface dreams in chat UI (with dream-diff artifact) *(diff 3, ~1w)*
Dreams appear as their own session kind (`metadata.kind="dream"`). Surface the **diff** ("yesterday's dream changed these 12 memories") as a structured artifact, not just narrative content. Read-only.
**Source**: `dreaming-research.md` Part IV §D open question §5; `dreaming-anthropic.md`.
**Depends on**: P0.

### P7 — User veto / edit on dream output *(diff 3, ~1w)*
Builds on P6. Let user accept/reject individual proposed memories. Rejected ones go to `status=contradicted` with `provenance=user_rejected:{ts}` and feed back into the dream's "don't propose this again" filter.
**Source**: `dreaming-research.md` Part IV §C.
**Depends on**: P6.

### P8 — Cross-scope insight discovery *(diff 4, ~2w)*
A second dream pass that *deliberately* looks across scopes for transferable patterns. Risky for scope leakage; needs careful prompting and a confirmation step before writing back.
**Source**: `dreaming-spec.md` §6.
**Depends on**: P0, P2.

### P9 — Lucid / user-directed dreams + directed daydreaming *(diff 3, ~1.5w)*
Two related modes:
- **Lucid dream (scheduled, deferred)** — user button "think about X tonight." Parameterized dream pass with user-supplied focus; runs at the next scheduled dream slot or whenever the batch queue flushes. Cheap (uses batch path, 50% off).
- **Directed daydreaming (live, on-demand)** — *awake-state* analog. User in the middle of a chat says "go daydream about this" (or the agent decides to self-invoke when stuck); a focused mini-dream-pass runs immediately, synchronously, with the user's stated prompt as the constraint frame. Output streams back into the same chat session as a structured "daydream" message — proposed memories, recombinations, weak-link discoveries. **Skips the batch path** (latency > cost here); uses the standard sync chat executor with a dream-tier system prompt and the restricted dream tool surface.

Both reuse `execute_dream_pass`; the daydream variant just sets `mode="daydream"` instead of `mode="dream"` and forces sync execution. Memories proposed by either land as `tentative` and ride the ratification loop (P0.4).
**Source**: `dreaming-research.md` Part II (lucidity); `dreaming-scheduler.md` §5; daydreaming as the natural awake-state pair of dreaming.
**Depends on**: P0, P6.

**Required prereq for daydreaming: inline `DreamOperations` on the chat stream.**
Today the dream-pass `DreamOperations` payload is only returned synchronously from the admin endpoint (`POST /api/admin/memory/{user_id}/dream`); the chat SSE stream (`POST /api/chat/sessions/{id}/stream`) does NOT emit dream events. Daydreaming needs that — the user-visible "go daydream about this" UX has to render the proposed memories / recombinations as a structured `daydream` message inline in the active chat, not behind an admin call. **Concrete change when P9 lands:** dream/daydream orchestrator emits a `dream.operations` SSE event on the active session's stream (gated on `mode in {dream, daydream}`); chat-stream frontend renders it as a "daydream summary" artifact. This is the same plumbing we'd want for P6 (Surface dreams in chat UI) so build them together. Tracked separately from P0.6 — eval-time we capture `DreamOperations` via the admin endpoint instead, no chat-stream change required.

### P10 — Affective recalibration (the proper version) *(diff 3, ~1w)*
Distinct from P0.3 fact-demotion. The Walker/van der Helm "overnight therapy" analog — re-rate `confidence` on memories tied to user frustration or strong emotion, preventing the agent from over-weighting one-off venting.
**Source**: `dreaming-research.md` Part II (overnight therapy).
**Depends on**: P0.

### P11 — Threat rehearsal / failure pre-bake *(diff 4, ~2w)*
Revonsuo-style — generate plausible failure scenarios for the user's active agents/workflows, pre-compute graceful-recovery paths, cache as procedure memories.
**Source**: `dreaming-research.md` Part II (threat simulation).
**Depends on**: P0, P1.

### P12 — Schema integration / multi-episode summaries *(diff 5, ~3w+)*
Properly hard. Durable consolidated-summary nodes that survive across many dream passes with stable identity, so "what I know about Project X" condenses without losing provenance. Hippocampus→neocortex analog; most ambitious memory-system change in this list.

**Status**: deep research in flight — see `dream/dreaming-multi-episode.md` (50M-token budget research pass). Wait for that doc before committing to a design. The research agent will survey Graphiti's unused `:Community` nodes, MemGPT/Letta, LangChain/LlamaIndex long-term-memory patterns, and propose a schema + build plan.

**Source**: `dreaming-research.md` Part II (memory consolidation); `dreaming-memory.md` §8 Priority 4.
**Depends on**: P0, P2, P1 (procedures).

---

## Deferred — with rationale

- **Cross-user community memory** — privacy nightmare for individual users, but **becomes natural once we have organizations**. Within an org, "the workflow your colleague figured out" is a valid memory share. Revisit when org-scoping lands in the platform; treat as P0 for org-scoped memory when that happens.
- **Pure creative recombination as a user-facing surface** (Cai/Wagner-style weird-connection-of-the-day) — not core value, but **could be a good whimsy / delight feature**. "Your dream made an unexpected connection between X and Y" as a daily card. Low-stakes, no memory-write side effects. Park as a stretch idea; consider after P7 once the dream surfacing UI exists.
- **Welfare-as-headline product framing** — premature; ship the engineering wins first, talk about welfare second.

---

## Cadence assumption

P0 now contains seven items (batch + recombination + two demotion modes + ratification + web fact-check + benchmark harness) — it is the full dream-system v1 plus measurement. Expect P0 to take ~8–10 engineer-weeks. P1 (procedures + skill-export) is day-two and lands after the procedure research doc is in. P2–P5 follow in the next month; P6–P7 the month after; P8+ are quarter-scale. Re-evaluate ordering after each item lands.

### P0 build-readiness gates

1. ~~`dream/dreaming-graphiti.md`~~ **DONE.** Audit fixes moved to **P-1** (above) — that PR is the base of the chain. P0 builds on top.
2. **P-1 PR merged into `dev`** — adds custom edge types, splits soft-delete, enables communities.
3. **PR [#12993](https://github.com/Significant-Gravitas/AutoGPT/pull/12993)** (local-LLM transport) — still gating. P0 architecture must support all four transports. See `p0-spec.md` §13.

---

## Product / non-engineering decision points

1. **Granularity** — one dream pass per user, or per `(user × scope)`? *Recommend: per-user for P0; revisit on metrics.*
2. **NREM vs REM phases** — one combined LLM call or two-phase (consolidate → recombine)? *Recommend: two-phase per `dreaming-spec.md` §1.*
3. **Welfare framing** — is dreaming for the agent or for the user? *Recommend: lead with user-value; welfare secondary.*
4. **Cost surfacing** — do batch savings show as a line item or absorbed as margin?
5. **ZDR tenants** — batch is not ZDR-eligible. Opt-out per-tenant; coordinate with legal.
6. **Skills as a platform concept** — P1 raises the question whether "Skill" deserves to be a first-class object alongside Agent and Block, or stays inside `ProcedureMemory`. Research doc will recommend; product owns the call.

---

## Active research passes

| File | Budget | Focus |
|---|---|---|
| `dream/dreaming-procedures.md` | 5M tokens | Pattern detection in chat → `ProcedureMemory` → user "Save as Skill" tool. Tilt toward uploadable skills and skills-as-first-class-platform-objects. |
| `dream/dreaming-multi-episode.md` | 50M tokens | Multi-episode consolidated summaries with stable identity. Survey Graphiti's `:Community`, MemGPT/Letta, LangChain/LlamaIndex, Anthropic Dreams memory model. Schema proposals + build plan. |
