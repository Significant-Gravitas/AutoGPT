# Anthropic Managed Agents *Dreams* — Reference Notes

A close read of the primary sources (Anthropic docs, blog, SDK examples, third-party reporting from launch week May 6–10 2026) so we can copy what's good and diverge where we have to. Companion to `dreaming-research.md` Part III §1.

---

## 1. The exact API contract

All endpoints sit under `https://api.anthropic.com/v1/`. Every request needs **two** beta headers (the SDK adds them automatically):

```
anthropic-version: 2023-06-01
anthropic-beta: managed-agents-2026-04-01,dreaming-2026-04-21
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/dreams` | Start a dream (returns `pending`) |
| `GET` | `/v1/dreams/{id}` | Poll status |
| `POST` | `/v1/dreams/{id}/cancel` | Cancel `pending`/`running`; 400 on terminal |
| `POST` | `/v1/dreams/{id}/archive` | Archive a terminal dream (no unarchive) |
| `GET` | `/v1/dreams?limit=20&include_archived=true` | List, newest first, paginated |

Request body for `POST /v1/dreams`:

```json
{
  "inputs": [
    { "type": "memory_store", "memory_store_id": "memstore_01Hx..." },
    { "type": "sessions", "session_ids": ["sesn_01...", "sesn_02..."] }
  ],
  "model": "claude-opus-4-7",
  "instructions": "Focus on coding-style preferences; ignore one-off debugging notes."
}
```

Full `dream` resource:

```json
{
  "type": "dream",
  "id": "drm_01AbCDefGhIjKlMnOpQrStUv",
  "status": "pending",
  "inputs": [ ... ],
  "outputs": [],
  "model": { "id": "claude-opus-4-7" },
  "instructions": "...",
  "session_id": null,
  "created_at": "2026-04-29T17:04:10Z",
  "ended_at": null,
  "archived_at": null,
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  },
  "error": null
}
```

`status` enum: `pending` → `running` → `completed` / `failed` / `canceled`. `outputs[]` is empty until `running`, at which point one entry appears: `{ "type": "memory_store", "memory_store_id": "memstore_..." }`. No webhooks — clients poll. `session_id` populates once `running` and points at an underlying Managed-Agents session you can stream live to watch the pipeline read/write the output store.

Error types observed in docs: `timeout`, `internal_error`, `memory_store_org_limit_exceeded`, `input_memory_store_too_large`, `input_memory_store_unavailable`, `input_session_unavailable`.

## 2. The data model

A **memory store** (`memstore_...`) is a workspace-scoped, hierarchical directory of plain-text "memory" documents (≤100 kB each, ~25 k tokens). Each memory has a path (`/preferences/formatting.md`), content, and a `content_sha256` for optimistic concurrency. Every mutation creates an immutable `memver_...` **memory version** (30-day retention, more for cold files). Stores can be archived (one-way, makes them read-only) or deleted.

A **session** (`sesn_...`) is one run of an agent inside a container. Memory stores are mounted into the session at `/mnt/memory/<store>/` and the agent reads/writes them with normal file tools at `read_only` or `read_write`. Up to 8 stores per session.

A **dream** (`drm_...`) is an async job whose input is `(memory_store, ≤100 sessions)` and whose output is **a new memory store**. The input store is never mutated — *"The input store is never modified, so you can review the output and discard it if you don't like the result."*

Memories are not strongly typed in Anthropic's world. There's no `memory_kind`, no `status`, no `confidence`, no `source_kind`. Just text at a path. Scoping is done by **multiple stores** (one per user, one per team, one for shared reference) attached to a session — there's no within-store namespace beyond directory paths.

## 3. The execution model

Dreams are **async batch**: minutes to tens of minutes depending on input size. They run on a managed pipeline; a `session_id` appears once `running` so you can watch the pipeline as an ordinary streaming session. No callback/webhook — poll `GET /v1/dreams/{id}` (the docs show a 10-second sleep loop). Default workspace rate limits apply during the research preview; contact support for higher.

Concurrency: nothing in the docs says you can't run multiple dreams simultaneously, and the list endpoint suggests a workspace can have many in-flight. **Constraint**: while a dream is `pending`/`running`, deleting or archiving its *output* store returns 400; archiving/deleting any *input* (the source store or any of the session transcripts) fails the dream with `input_memory_store_unavailable` / `input_session_unavailable`. So the inputs are effectively pinned for the duration.

Termination: `completed` (use `outputs[0].memory_store_id`), `failed` (output store persists with partial contents), `canceled` (same — output kept for inspection). Cancel is idempotent on already-canceled; 400 on terminal.

## 4. Immutability and diffability

Input memory store is never mutated. The output is a **separate, fully-populated store** that lives in the workspace like any other. To "promote" it you re-create your sessions with the new `memory_store_id` in `resources[]` — or attach both old and new in parallel. To discard, call `memory_stores.delete` or `memory_stores.archive`. **There is no diff endpoint** — Anthropic ships the two stores side by side and leaves diffing to the caller. The Memory Stores API supports listing memories by `path_prefix`, retrieving content, and listing `memory_versions` per store, so a diff is `list + retrieve` on both stores client-side. Memory versions give you per-memory audit history, including a `redact` operation for compliance.

## 5. What goes IN the prompt

The dream API exposes exactly one tunable: **`instructions`**, ≤4,096 characters, free-form natural language. The published example is one line:

> *"Focus on coding-style preferences; ignore one-off debugging notes."*

The implicit prompt is opaque. Anthropic only tells you the mechanism in prose: *"reviews your agent sessions and memory stores, extracts patterns, and curates memories so your agents improve over time"* — *"duplicates merged, stale or contradicted entries replaced with the latest value, and new insights surfaced."* No knobs for temperature, no system-prompt override, no choice of NREM-vs-REM-style phases, no way to specify "only touch this subset of paths." You pick the model (`claude-opus-4-7` or `claude-sonnet-4-6`) and give it an English-language steer; the pipeline does the rest.

## 6. Customer outcomes reported

- **Harvey (legal AI)**: *"completion rates went up ~6x in their tests"* after adding dreaming. No baseline, no task description, no time window. This is the only dreams-specific number Anthropic publicly cites.
- **Wisedocs (medical document review)**: *"cut its document review time by 50%"* — but this is an **outcomes** result, not dreams.
- **Outcomes evals**: *"improved task success by up to 10 points over a standard prompting loop, with the largest gains on the hardest problems"* — overall. File-gen wins: +8.4% docx, +10.1% pptx. Again, outcomes-specific.
- **Netflix**: cited as a multi-agent orchestration early adopter, not dreams.

Press framing (VentureBeat, SiliconANGLE, The New Stack) consistently calls dreams an "offline replay → consolidate → short-to-long-term memory" analogue, but **Anthropic's own copy never invokes neuroscience**. Alex Albert's published quote in VentureBeat is the organizational analogy: "*analogous to people in an organization writing up skills after a task.*"

## 7. Pricing and limits

- Billed at **standard model token rates** for whichever model you select. The `usage` field on the dream resource reports `input_tokens / output_tokens / cache_creation_input_tokens / cache_read_input_tokens`. *"Cost scales roughly linearly with the number and length of input sessions."*
- One third-party explainer (buildfastwithai) cites **$0.08 per agent runtime hour** for the broader Managed Agents harness on top of token costs — that's the platform charge, not a dreams-only line.
- **Limits**: 100 sessions per dream; 4,096-char `instructions`; supported models `claude-opus-4-7` and `claude-sonnet-4-6`; default workspace rate limits during preview.
- Memory: ≤100 kB per memory file, ≤8 stores per session, 30-day version retention.

What's expensive: the pipeline ingests every transcript in `session_ids[]` plus the full input store, then writes a brand-new store. With Opus 4.7 input/output rates and up to 100 transcripts that can easily be millions of tokens per pass — so the docs explicitly tell you to *"start with a small batch of sessions and scale up once you're satisfied with the curation quality."*

## 8. Where Anthropic's design fails our use case

1. **Memory model mismatch.** Their store is a directory of flat text files. Ours is a temporal graph (Graphiti/FalkorDB) with entities, episodes, semantic-fact edges. We can't "produce a new memory store" — we'd produce a new *graph snapshot* or a *proposed-changes set*.
2. **No typing.** Anthropic has zero structure beyond paths. We have `MemoryEnvelope` with `memory_kind ∈ {fact, preference, rule, finding, plan, event, procedure}`, `source_kind ∈ {user_asserted, assistant_derived, tool_observed}`, `status ∈ {active, tentative, superseded, contradicted}`, `confidence`, `provenance`, `scope`. A dream pass for us should *write into these types*, not bypass them — the `MemoryStatus` ladder (especially `tentative` → `active` ratification) was apparently built for exactly this lifecycle.
3. **Developer-only audience.** Dreams is API-only with a request-access form. Our chat product is end-user-facing; we want users to see/veto/feel ownership of their dream output. Anthropic's framing ("operator runs the pipeline, reviews the new store, attaches or discards") is wrong for consumer-grade chat.
4. **Manual trigger.** Anthropic has no built-in scheduler — *you* call `POST /dreams` when you want one. We already have APScheduler with per-user cron, and the natural cadence is user-3am nightly, automatic. Don't ship a manual button as the only entrypoint.
5. **Scope = one store.** Anthropic ties a dream to one memory store. Our scopes are richer (`real:global`, `project:*`, `book:*`, `session:*`). We probably want per-scope dreams so that "book/Hamlet" and "project/billing-rewrite" consolidate independently.
6. **No phase distinction.** One opaque pipeline, one model. The neuroscience literature and the existing ML "dreaming" tradition (Dreamer, Sleep-time Compute, SCM) all suggest two phases — verbatim consolidation (cheap) and creative recombination (expensive). We can split.
7. **No measurement hooks.** Anthropic gives you `usage` (tokens) and that's it. Was the dream *useful*? We need post-dream warm-context relevance, ratification rate, contradiction-detection rate, sleep-time-compute-style cache-hit rate — none of which Anthropic exposes.
8. **Output is a *replacement*.** Theirs is take-it-or-leave-it: attach the new store or throw it away. Ours should be *additive proposals* against a graph, gated by `status=tentative` so wake-time chat ratifies or contradicts. That's both safer and lines up with how Graphiti edges already work.

## 9. Where to copy verbatim

- **Immutable input.** Whatever the dream reads, it must not mutate. Snapshot the relevant Graphiti subgraph; write outputs as a new ChatSession/episode set tagged `dream`.
- **Diffable output.** Produce a discrete, addressable artifact (a "dream session" with `metadata.kind=dream`) so users can inspect, accept, or roll back. Mirror the version-history pattern with our existing `MemoryStatus.superseded`.
- **Async batch with poll.** Don't try to stream the whole pipeline interactively. Fire-and-forget through RabbitMQ/APScheduler, expose a single status endpoint, optionally pipe live events into the existing SSE channel for users who want to watch.
- **`instructions` field.** A ≤4 k-char natural-language steer is exactly the right shape for letting users say "weight my reading notes higher than my morning rants." Add it to `ChatSessionMetadata.dream_config`.
- **"Extract patterns from sessions" framing.** Anthropic doesn't claim creativity, doesn't invoke neuroscience in copy, doesn't promise consciousness. Just: "we found patterns across your sessions; here they are." Copy that humility — it ships.
- **Graceful failure semantics.** On `failed`/`canceled`, keep the partial output for inspection rather than rolling back. That's a good ergonomic default.
- **Cancel + archive separation.** `cancel` for in-flight, `archive` for terminal. Clean state machine, easy to model in APScheduler.

---

## 10. Anthropic does X / We should do Y

| Dimension | Anthropic Dreams | Us |
|---|---|---|
| **Trigger** | Manual `POST /dreams` per call | APScheduler cron per user at user-local 3am, `max_instances=1`, job-id `dream_pass_{user_id}` |
| **Audience** | Developer / API operator | End user, surfaced as a `metadata.kind=dream` ChatSession in the existing copilot UI |
| **Memory substrate** | Flat directory of text files in a `memory_store` | Temporal graph in Graphiti/FalkorDB; episodes + semantic edges, no document layer |
| **Memory typing** | None — paths only | `MemoryKind` × `SourceKind` × `MemoryStatus` × `confidence` × `scope` from `memory_model.py` |
| **Output shape** | A whole new `memory_store` resource | New episodes via `enqueue_episode()` with `source_kind=assistant_derived`, `status=tentative`, plus `status=superseded` flips on contradicted facts |
| **Acceptance model** | Op attaches new store or deletes it | User-ratified at wake time: `tentative` → `active` on next-session reuse, `contradicted` if user pushes back |
| **Scope** | One store per dream | Per-(user × scope) so `project:*`, `book:*`, `real:global` dream independently |
| **Phases** | One opaque pipeline | Two-phase: cheap consolidation on Sonnet (NREM-equivalent) then creative recombination on Opus (REM-equivalent) — gives us a knob for cost |
| **Tunables** | `instructions` ≤4096 chars, model pick | Same `instructions` field + per-scope toggle + `dream_enabled` per user, all in `ChatSessionMetadata` JSON (no migration) |
| **Observability** | `usage` tokens only | Warm-context relevance, ratification rate, contradiction-detection rate, cache-hit on next 24 h queries — instrumented day one |
| **Failure mode** | Partial output store left around | Dream session marked `metadata.dream_status=failed` with partial episodes still attributed but flagged; user can re-trigger |
| **Pricing exposure** | Pure token pass-through | Same upstream, but we abstract behind a per-user `dream_quota` so heavy users don't surprise-bill |
| **Welfare framing** | None — pure memory hygiene | Optional: expose the dream session content so users feel ownership; copy stays product-focused, not mystical |
