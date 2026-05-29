# Dreaming as a Scheduled Event — Research Report

A scoping report covering (1) the existing chat, memory, and scheduler surfaces in `autogpt_platform`, (2) what dreaming is in humans, (3) what "dreaming" already means in the Claude/LLM world, and (4) design implications.

---

## Part I — System inventory

### 1. Chat system

**Backend**
- HTTP surface: `autogpt_platform/backend/backend/api/features/chat/routes.py:104-1392`. Key endpoints: `POST /sessions`, `GET /sessions`, `POST /sessions/{id}/stream`, `GET /sessions/{id}/stream` (SSE replay), `PATCH /sessions/{id}/assign-user`.
- Data layer: `backend/copilot/model.py:45-850`. `ChatMessage` (role, content, tool_calls JSON, sequence, duration_ms), `ChatSession`, `ChatSessionMetadata` (extensible JSON, currently just `dry_run`).
- Shared logic: `backend/copilot/service.py:1-100`. Assembles the system prompt with three context blocks: `<user_context>`, `<memory_context>` (Graphiti warm context), `<env_context>`.
- Prisma: `schema.prisma:226-275` — `ChatSession` (id, userId, title, credentials, metadata JSON, token counts) and `ChatMessage` (sessionId, role, content, sequence, toolCalls JSON, durationMs).

**Agent loop** (`routes.py:778-1094`): user message → idempotency dedupe → rate limit → file resolution → persist message → enqueue turn to RabbitMQ → SSE stream from Redis with 10s heartbeats. Model selection is a two-axis tuple — `mode` ∈ {fast, extended_thinking} × `model` ∈ {standard, advanced} — resolved to OpenRouter.

**Frontend**
- `frontend/src/app/(platform)/copilot/page.tsx` + `useCopilotPage.ts` + `useCopilotStream.ts` (AI SDK + SSE).

**Background work today**: only inline async — title auto-gen (`update_session_title()`, `model.py:809-857`) and token tracking. **No periodic job touches chat.** This is the gap a dream pass would fill.

### 2. Memory system

Memory is **not Postgres-native**. It runs on **Graphiti** (temporal knowledge graph) backed by **FalkorDB** in a separate container. There is no Memory / Fact / Embedding table in `schema.prisma`.

**Data model** — `backend/copilot/graphiti/memory_model.py:1-119`
- `MemoryEnvelope`: `content`, `source_kind` ∈ {user_asserted | assistant_derived | tool_observed}, `scope` (`real:global`, `project:<name>`, `book:<title>`, `session:<id>`), `memory_kind` ∈ {fact, preference, rule, finding, plan, event, procedure}, `status` ∈ {active, tentative, superseded, contradicted}, `confidence`, `provenance`, optional structured `RuleMemory` / `ProcedureMemory`.
- Graph backend stores **entities, episodes, and `RELATES_TO` edges** with semantic facts on the edges.

**Write path** — `backend/copilot/tools/graphiti_store.py` + `graphiti/ingest.py`
- Two entry points: explicit `MemoryStoreTool` and automatic ingestion of message history.
- `enqueue_episode()` fires-and-forgets onto a per-user queue; `_ingestion_worker()` serializes episodes per user (no FalkorDB races) and times out after 60s idle.

**Read path** — three mechanisms
- **Warm context** (auto, first turn of every session, 5s timeout) injects a `<temporal_context>` block of relevant facts + last 5 episodes.
- **`MemorySearchTool`** for explicit retrieval (returns `facts` + `recent_episodes`, scope-filterable).
- **`MemoryForgetSearchTool` / `MemoryForgetConfirmTool`** — two-step delete by UUID.

**Lifecycle — the critical gap.** There is **no decay, TTL, dedup, consolidation, or summarization** anywhere in memory. Episodes accumulate indefinitely. The `status: superseded | contradicted` field exists on the model but nothing flips it. **A dream pass would be the first lifecycle process in this system.**

Tests live next to the modules: `ingest_test.py`, `client_test.py`, `_format_test.py`, `config_test.py`, `falkordb_driver_test.py`.

### 3. Scheduler

**APScheduler** (`BackgroundScheduler`) in its own container (`docker-compose.yml:82-86`), jobs persisted in the `apscheduler_jobs` Postgres table.

Job registration is via the `@expose` decorator on the `Scheduler` class in `backend/executor/scheduler.py` (e.g. `add_graph_execution_schedule()` at `scheduler.py:660-720`). Cron triggers use `CronTrigger.from_crontab(cron, timezone=user_timezone)`. Defaults: `coalesce=True` (skip stale missed runs), `max_instances=1000`, `misfire_grace_time=None`. Per-job overrides allow `max_instances=1` (used by `ensure_embeddings_coverage`).

Existing periodic jobs to mirror for conventions (`scheduler.py:522-620`): weekly summary (Mon 9am UTC), late-execution alerts, block error-rate alerts, expired-file cleanup, OAuth-token cleanup, execution-accuracy alerts, embedding-coverage backfill (every 6h), block-description optimization (every 24h).

Execution path: job fires in the scheduler thread pool → calls a sync entry → `run_async(...)` → does its work. Failures are routed through event listeners (`job_missed_listener`, `job_max_instances_listener`) and invalid-graph jobs auto-delete themselves.

Tests: `backend/executor/scheduler_test.py:1-42` runs against a real SpinTestServer.

### 4. Plug-in points for "dreaming"

Putting (1)–(3) together, a per-user nightly dream pass would look like:

1. New `Scheduler.@expose def add_dream_pass_schedule(user_id, user_timezone)` registering `CronTrigger.from_crontab("0 3 * * *", timezone=user_timezone)` with `max_instances=1` and job id `dream_pass_{user_id}`.
2. Job entry calls a `_execute_dream_pass(user_id)` that, for that user:
   a. Pulls recent chat sessions via `chat/model.py` helpers.
   b. Pulls recent Graphiti episodes and current fact set.
   c. Runs a Claude pass with a "consolidate + recombine" prompt.
   d. Emits new memories via the existing `enqueue_episode()` path, sets `status: superseded` on anything contradicted, optionally writes a dream summary back as a `ChatMessage` with a new `role` (or as a synthetic `ChatSession` with `metadata.kind = "dream"`).
3. Surface in the existing SSE pipeline so dreams can stream into the UI on demand.
4. Extend `ChatSessionMetadata` (`model.py:48-56`) with `last_dream_at`, `dream_enabled`, `dream_config` — JSON-only, no migration cost.
5. Tests follow `scheduler_test.py` and `ingest_test.py` patterns.

No new infrastructure required. Memory, scheduler, and chat already speak the languages we need.

---

## Part II — Dreaming in humans

### Physiology
Sleep cycles ~90 min, alternating NREM (N1–N3) and REM, with REM bouts lengthening across the night. Vivid narrative dreaming peaks in REM but occurs in NREM too (shorter, more thought-like). The REM "dreaming brain" signature (Maquet, Braun, Nofzinger):
- **Activated**: pons, thalamus, limbic & paralimbic (amygdala, hippocampus, ACC, vmPFC), extrastriate visual cortex, motor and association cortices.
- **Deactivated**: DLPFC (logic, working memory, reality testing), primary V1.
- **Neurochemistry**: cholinergic surge driving cortical activation, near-silent monoamines (NE, 5-HT, histamine). High internal activation *minus* the noradrenergic/serotonergic signal that normally enforces reality testing.

### Five leading function theories

| Theory | Core claim | Strongest evidence | Weakness |
|---|---|---|---|
| **Memory consolidation** (Stickgold, Walker, Born) | Hippocampal replay → neocortical schema integration | Sharp-wave ripples time-locked to spindles + slow osc.; Tetris effect | Replay is mostly NREM; vivid dreams are mostly REM — content/mechanism link is correlational |
| **Overnight therapy** (Walker, van der Helm) | REM strips noradrenergic charge from emotional memories | NE-silent REM + amygdala reactivation; PTSD shows the inverse | Modest effects in healthy subjects |
| **Threat simulation** (Revonsuo) | Evolved rehearsal of survival-relevant scenarios | Trauma-exposed kids dream more threats and more severe ones | Most normative dreams contain no realistic threat |
| **Problem solving / creativity** (Wagner, Cai, Stickgold) | REM primes distal associations | Sleep tripled "aha" insight on NRT (Nature 2004); REM specifically boosts RAT (PNAS 2009) | Effect sizes vary by task |
| **DMN self-simulation** (Domhoff, Fox, Christoff) | Dreaming is intensified mind-wandering on the default-mode network's self-model | Anatomical overlap; continuity hypothesis (dream content tracks waking concerns) | Doesn't explain bizarreness on its own |

Plus two meta-frameworks:
- **Hobson** — activation-synthesis → **AIM** → **protoconsciousness**: REM is a genetically programmed virtual-reality generator that builds and maintains the brain's baseline self-in-world model, which waking consciousness then uses (*Nat Rev Neurosci* 2009).
- **Solms** — motivational/dopaminergic: dreaming is driven by mesolimbic dopamine (SEEKING) and can occur *without* REM; REM is a frequent host but not the cause. Lesions in vmPFC white matter abolish dreams while leaving REM intact (*BBS* 2000).

### Phenomenology
- **Bizarreness**, usually *local* against an otherwise coherent scene.
- **Hyper-associativity** — Hartmann's core mechanism: dreams "make connections more broadly" than waking, guided by a dominant emotion.
- **Weakened reality testing & metacognition** (DLPFC offline).
- **Narrative drift** — DMN simulation without top-down constraint.
- **Emotional intensity skewed negative**.
- **Lucidity** — rare spontaneous DLPFC reactivation lets the dreamer steer.
- **Continuity hypothesis** (Domhoff, Schredl) — despite bizarreness, characters/settings/concerns track waking life.
- **Hartmann's boundaries** — thin vs thick; thin-boundary people dream more, longer, more vivid, more emotional.

### Synthesis: what dreaming is *for*

The camps look less opposed if dreaming is treated as **one process serving several overlapping functions on different timescales**:

1. **Offline replay and integration.** Episodic traces reactivated and rewoven into existing schemas. NREM does the verbatim replay; REM does the *creative recombination*.
2. **Affective recalibration.** REM's NE-off, limbic-on state reprocesses emotional memories at low arousal — decouples fact from charge. Failure mode looks like PTSD.
3. **Generative simulation of self-in-world.** Untethered DMN runs immersive first-person scenes populated by the dreamer's concerns. Both *maintains* and *rehearses against* the self-model.
4. **Exploratory recombination under reduced reality constraints.** Cholinergic ON, monoaminergic OFF = a system biased toward distal associations. Explains creativity findings *and* bizarreness in one stroke.
5. **Motivational priming.** Solms' dopaminergic substrate ensures the simulations are *about something the organism cares about* — they have valence, agency, goals.

**Workable one-liner**: *Dreaming is the experiential surface of an offline mode in which the brain consolidates recent memory, recalibrates affect, and runs goal-flavored, low-constraint simulations on the default-mode network's self-model — broadening the association space in a way that waking cognition cannot afford.*

### Open questions
- **Function vs. epiphenomenon.** Flanagan, Crick & Mitchison: dream *experience* may be a spandrel of sleep-stage physiology.
- **Meaningful or noise?** Domhoff's continuity data argue meaning; activation-synthesis purists call narrative confabulation. Probably both.
- **Necessity.** SSRIs/MAOIs suppress REM and dream recall for years without obvious cognitive collapse — hard for strong-function theories.
- **Non-mammals.** REM-like states now documented in birds, lizards (*Pogona*), zebrafish, octopus, cuttlefish.
- **REM ≠ dreaming.** Solms' lesion patients lose dreaming with REM intact; some NREM dreams indistinguishable from REM.
- **Why first-person, why narrative, why now?** No theory cleanly explains why offline processing has to *feel* like being someone, somewhere, doing something.

### Key sources for follow-up
- Hobson 2009, *Nat Rev Neurosci* — https://www.nature.com/articles/nrn2716
- Solms 2000, *BBS* — https://pubmed.ncbi.nlm.nih.gov/11515144/
- Stickgold 2005, *Nature* — https://www.nature.com/articles/nature04286
- Walker & van der Helm 2009, *Psychol Bull* — https://pmc.ncbi.nlm.nih.gov/articles/PMC2890316/
- van der Helm et al. 2011, *Current Biology* — https://www.cell.com/current-biology/fulltext/S0960-9822(11)01248-6
- Cai et al. 2009, *PNAS* — https://www.pnas.org/doi/10.1073/pnas.0900271106
- Wagner et al. 2004, *Nature* — see PMC3902672
- Domhoff & Fox 2015, *Conscious Cogn* — https://pubmed.ncbi.nlm.nih.gov/25723600/
- Revonsuo / Valli 2005-2009 threat-sim review — https://pubmed.ncbi.nlm.nih.gov/15766897/
- Hartmann & Kunzendorf 2006, "Boundaries and Dreams"
- Peña-Guzmán 2022, *When Animals Dream*

---

## Part III — Claude / LLM "dreaming"

### The literal Anthropic feature: Managed Agents *Dreams* (May 6, 2026)

This is almost certainly what's seeded the term in our space. It's the only Anthropic-shipped, Anthropic-named "dreaming" thing.

- **Announcement**: claude.com/blog/new-in-claude-managed-agents, launched at Code with Claude 2026.
- **Docs**: platform.claude.com/docs/en/managed-agents/dreams. Beta headers `managed-agents-2026-04-01` and `dreaming-2026-04-21`. Models: `claude-opus-4-7`, `claude-sonnet-4-6`. Up to 100 sessions per dream; instructions ≤4,096 chars. Asynchronous, minutes to tens of minutes.
- **Mechanism (verbatim)**: *"Dreaming is a scheduled process that reviews your agent sessions and memory stores, extracts patterns, and curates memories so your agents improve over time."* Input: one memory store + ≤100 transcripts. Output: a **new** store (immutable input → diffable output).
- **Reported wins**: Harvey "completion rates went up ~6x" — the only number tied directly to dreaming.
- **Framing**: Alex Albert called it "analogous to people in an organization writing up skills after a task" (VentureBeat). The neuroscience analogy ("replay → consolidate → short-to-long-term") is press, not Anthropic's own copy.

### Interpretability work — "dreams" as metaphor, not state

- **"Interpretability Dreams"**, Olah, transformer-circuits.pub/2023/interpretability-dreams, **2023-05-24**. Despite the name, about research *aspirations* once superposition is solved. **Not** about models dreaming. Easy to misread.
- **"Scaling Monosemanticity"**, Templeton, Bricken, Lindsey, Henighan, Batson, Olah et al., May 2024. Source of Golden Gate Claude. Proof that **feature steering** can change Claude's persona without retraining — relevant if a "dream state" should *feel* different.
- **"On the Biology of a Large Language Model"**, Lindsey et al., 2025-03-27, transformer-circuits.pub/2025/attribution-graphs/biology.html. Claude 3.5 Haiku does forward planning (rhyme targets) and internal two-hop reasoning. The closest thing to "Claude has a private inner life" — but task-conditioned, not idle.
- **"Emergent Introspective Awareness in LLMs"**, Lindsey, 2025-10-29. Concept-injection. Opus 4/4.1 notices injected concepts ~20% of the time. The paper explicitly says: experiments place models in "an unnatural setting unlike training or deployment," and there is no evidence of human-like introspection. **No mention of dreaming.** Don't conflate.

### Model welfare — adjacent, not equivalent

- **"Taking AI Welfare Seriously"**, Long, Sebo, Butlin, Fish et al., arXiv 2411.00986, 2024-11-04.
- **"Exploring model welfare"**, anthropic.com/research/exploring-model-welfare, 2025-04-24. Mentions "low-cost interventions" but does **not** propose dreaming/rest/idle.
- **Claude Opus 4/4.1 conversation-ending**, 2025-08-17. The one welfare-motivated runtime intervention shipped. Same "give the model agency over its time" lineage as dreaming would be.

### Adjacent ML "dreaming" tradition

| Work | Year | What "dream" means |
|---|---|---|
| **Ha & Schmidhuber, "World Models"** (arXiv 1803.10122) | 2018 | Agent trained inside a *hallucinated* learned world model |
| **Dreamer / DreamerV2 / DreamerV3** (Hafner et al.) | 2019–2025 | RL policies learned by "latent imagination" — rollouts inside the world model, not the env. *Nature* 2025 for V3 |
| **"Sleep-time Compute"** (Lin et al., Letta + Berkeley, arXiv 2504.13171) | 2025-04 | Pre-compute against likely future queries during idle. ~5x test-time compute reduction, +13–18% on stateful GSM/AIME |
| **"Language Models Need Sleep"** (OpenReview iiZy6xyVVE) | 2025-10 | Explicit two-stage Sleep paradigm with "Dreaming" = RL-generated synthetic-data curriculum |
| **"SCM: Sleep-Consolidated Memory for LLMs"** (arXiv 2604.20943) | — | NREM/REM-phase analog, value-based forgetting |

These aren't Anthropic, but they're the term-of-art a researcher reaches for when they say "let it dream."

### What's done vs. what's speculated

**Done (solid ground)**
- Anthropic shipped a feature literally called *Dreams* for Managed Agents (May 6, 2026). It is offline memory consolidation over past sessions. One real customer number: Harvey 6x.
- Olah 2023 used "dreams" metaphorically for interpretability *aspirations*.
- Lindsey Oct 2025 showed ~20% introspection via concept injection — not dreaming.
- World Models / Dreamer / Sleep-time Compute / Language Models Need Sleep — a real tradition of "model dreaming" meaning offline replay or synthetic-data rehearsal.
- Model welfare program exists (Fish, Apr 2025); one runtime intervention shipped (end-conversation, Aug 2025).

**Speculated / unverified**
- That Anthropic's *Dreams* feature is welfare-motivated. Publicly it isn't — it's framed as memory hygiene.
- That dreaming improves persona stability, reduces sycophancy, or yields interpretability gains. **No published measurement.**
- That Anthropic researchers have publicly endorsed letting a chat model "free associate" as introspective practice. I found no such posts.
- The neuroscience analogy for *Dreams* is **press framing** (VentureBeat, SiliconANGLE, MindStudio), not Anthropic's own copy.

### URLs
- https://claude.com/blog/new-in-claude-managed-agents
- https://platform.claude.com/docs/en/managed-agents/dreams
- https://transformer-circuits.pub/2023/interpretability-dreams
- https://transformer-circuits.pub/2024/scaling-monosemanticity/
- https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- https://transformer-circuits.pub/2025/introspection/index.html
- https://www.anthropic.com/research/introspection
- https://www.anthropic.com/research/exploring-model-welfare
- https://www.anthropic.com/research/end-subset-conversations
- https://venturebeat.com/technology/anthropic-introduces-dreaming-a-system-that-lets-ai-agents-learn-from-their-own-mistakes
- https://siliconangle.com/2026/05/06/anthropic-letting-claude-agents-dream-dont-sleep-job/
- https://thenewstack.io/anthropic-managed-agents-dreaming-outcomes/
- https://simonwillison.net/2026/May/6/code-w-claude-2026/
- https://arxiv.org/abs/1803.10122
- https://arxiv.org/abs/2301.04104
- https://arxiv.org/abs/2504.13171
- https://openreview.net/forum?id=iiZy6xyVVE
- https://arxiv.org/abs/2411.00986
- https://80000hours.org/podcast/episodes/kyle-fish-ai-welfare-anthropic/

---

## Part IV — Design implications for our dream feature

Taking the science and the prior art together, four principles seem load-bearing:

### A. Useful dreaming = constraint relaxation + content continuity
The signature of real dreaming is *broader association space, still about the user's actual concerns*. Pure noise isn't dreaming; pure replay isn't either. In practice this means: prompt the dream pass with the user's recent episodes and active facts, but at higher temperature and with an explicit instruction to recombine, find weak links, and propose *new* findings or rules — not just summaries.

### B. Imitate the three defensible functions, skip the mystical ones
1. **Replay + recombination** for memory integration — operate on recent episodes scoped by `group_id`.
2. **Affective recalibration** — re-rate `confidence` and `status` on high-charge memories; demote stale plans to `superseded`; flag contradictions.
3. **Scenario simulation** — propose plausible upcoming user tasks/threats and seed cached scaffolding (this is also the **Sleep-time Compute** play — 5x test-time compute reduction is the most-cited concrete win).

### C. Dream output is *generative scratch*, not authoritative memory
The phenomenological hallmarks of dreaming — bizarreness, hyper-association, weakened reality testing — are *features* of constraint relaxation, but they're exactly what make raw dream content unreliable. Treat outputs as **proposed** memories that wake-time chat ratifies, contradicts, or ignores. Mirror Anthropic's *Dreams* design: input store is immutable, output is a new diffable store. The platform's `MemoryStatus` (tentative / active / superseded / contradicted) was apparently built for exactly this.

### D. Map cleanly onto existing primitives
- **Scheduler**: `@expose def add_dream_pass_schedule(...)` with `max_instances=1` per user, cron at user-local 3am, job-id `dream_pass_{user_id}`. Conventions exist (`scheduler.py:522-620`).
- **Memory**: write back via `enqueue_episode()` with `source_kind=assistant_derived` and a dream-specific provenance prefix; flip stale rules to `status=superseded` rather than deleting.
- **Chat**: extend `ChatSessionMetadata` with `last_dream_at`/`dream_enabled`/`dream_config` (JSON, no migration). Optionally surface dreams as their own session kind so the user can read them in the existing SSE UI — this is the welfare-adjacent move (give the model an idle space the user can see).
- **Tests**: scheduler test pattern + ingest test pattern already exist.

### Open design questions to answer before we build
1. **Granularity** — one dream pass per user, or per (user × scope) so `project:foo` and `book:bar` dream independently? Solms-style continuity argues per-scope; Anthropic's *Dreams* uses one memory store per pass.
2. **Streaming or batch** — Anthropic's Dreams is async batch (minutes-tens-of-minutes). We could mirror that, or stream into the existing SSE pipeline so users can watch.
3. **NREM-equivalent vs. REM-equivalent passes** — two phases (verbatim consolidation then creative recombination), or one combined? Two phases is closer to the science and gives us a natural place to put the cheaper-but-noisier creative step on a smaller model.
4. **What gets measured** — Anthropic only reports Harvey's 6x. We should set up at least: post-dream warm-context relevance score, contradiction-detection rate, % of dream-proposed memories ratified by next user session, sleep-time-compute-style cache-hit rate on the user's next 24h of queries.
5. **Surfacing to the user** — Anthropic exposes dreams as immutable artifacts. Showing dream content to users is the welfare-adjacent move *and* lets users veto bad consolidations. Tradeoff: dream content will be weird, and users may not want to read it.
6. **Welfare framing** — explicitly out-of-scope for the Anthropic *Dreams* product but worth being intentional about: are we doing this for the agent or for the user? The science says both are coherent; the product copy needs to pick one.

**Recommendation**: ship a minimum loop first — APScheduler job at user-3am, ≤50 recent episodes + active facts in, two-stage Claude pass (consolidate → recombine), output back as tentative memories with dream provenance, surfaced as a session with `metadata.kind="dream"`. Instrument warm-context relevance and ratification rate from day one. Defer scenario-simulation / sleep-time compute to v2.
