# Batch Processing as a First-Class Chat Path

Dreams are the obvious first consumer of Anthropic's Message Batches API — they're non-interactive, run overnight, and are pure cost. But the design pressure is broader: any non-streaming, latency-tolerant LLM call in the platform should be eligible for the 50% batch discount. This doc plans the integration.

## 1. Cost case

Anthropic's Message Batches API charges **50% of standard token prices** across all features (tool use, extended thinking, prompt caching, vision). For Opus 4.7 at standard list ($15/$75 per million in/out), batch is effectively $7.50/$37.50. Combined with the **1-hour prompt cache** (which is recommended and only meaningfully cheaper than the 5-minute cache when batches run longer than 5 minutes), a dream pass over 50 episodes with a stable system prompt looks like:

- 100k input tokens (system + memory + episodes), of which ~80k cacheable → ~$0.30 cache write + ~$0.024 cache hit reads after the first request
- 20k reasoning + output tokens → ~$0.75
- **Per-user-per-night: ~$0.30 with cache, ~$0.75 without batch discount, ~$1.50 on the synchronous Opus path**

At 10k users dreaming nightly, that's **~$3k/month batched vs. ~$15k/month synchronous**. Same math applies to any deferred chat or agent run.

## 2. API contract (verified from docs)

**Endpoints** (all under `https://api.anthropic.com/v1/messages/batches`):
- `POST /` — create a batch with up to 100k `requests`, each with `custom_id` + full Messages-API `params`
- `GET /{id}` — retrieve status; check `processing_status` ∈ {`in_progress`, `ended`, `canceling`}
- `GET /{id}/results` — stream JSONL once `ended`; per-request result types: `succeeded`, `errored`, `canceled`, `expired`
- `POST /{id}/cancel` — best-effort, may not stop already-in-flight items
- `GET /` (list) with `?limit=20&after_id=...` pagination
- `DELETE /{id}` after results expire

**Hard limits**:
- 100k requests OR 256 MB per batch (whichever first)
- 24-hour hard expiry; most batches complete in <1 hour
- `max_tokens >= 1` per request (no cache pre-warming inside a batch)
- Rate limits: separate token bucket for batches; see Anthropic's per-org limits
- **Not ZDR-eligible** — flagged in docs; matters for any enterprise data-handling claims we make

**Supported features inside a batch**: tool use, extended thinking, system prompts (cacheable), prompt caching (prefer 1h TTL), vision, structured outputs. **NOT supported**: streaming. Validation of each request's `params` is async — bad params show up as `errored` results, not at submit time.

**Statuses & retrieval**: response on create includes `results_url: null` and `expires_at`. Polling pattern:
```python
while batch.processing_status != "ended":
    await asyncio.sleep(30)
    batch = await client.messages.batches.retrieve(batch_id)
for result in client.messages.batches.results(batch_id):  # streams JSONL
    ...
```
No webhook in the core docs as of today; status is polled. (Anthropic has webhook beta work in flight but it's not the primary documented path — treat polling as the contract we build against.)

## 3. Where this plugs into our chat stack

We have two synchronous paths today, both ending at OpenRouter:

| Path | File | Today |
|---|---|---|
| baseline | `copilot/baseline/service.py:1-100` | OpenAI-compatible streaming via `_get_openai_client()` → OpenRouter |
| SDK | `copilot/sdk/service.py` | Claude Agent SDK subprocess → OpenRouter (`config.openrouter_active`) |

**OpenRouter does not currently expose Anthropic batches.** Batch traffic must go direct to `api.anthropic.com` using the Anthropic Python SDK. This means a third execution path — not "another mode in the existing services" — because the contract is async (no SSE), the auth is different (Anthropic key, not OpenRouter), and the message shape is the native Anthropic Messages API rather than OpenAI-compatible.

The right model is a **third sibling** to baseline/SDK: `copilot/batch/service.py`, driven by a new `mode = "batch"` literal in `CopilotMode`, with the existing `copilot_executor` keeping the synchronous paths and a new `copilot_batch_executor` service running the poll loop.

## 4. Three integration tiers (ship in order)

**Tier 1 — Dreams (no UX change).** A dream pass is the cleanest possible batch consumer: one user-initiated request, no streaming expectations, output is a memory diff. The scheduler enqueues a `dream_pass_requested` envelope; the batch executor accumulates these into a batch every N minutes (configurable, default 5), submits, polls, and on `ended` writes results back through the existing `enqueue_episode()` path. **Coalescing is the killer feature** — 10k user dreams across 24 timezones become roughly 24 batches/day, not 10k API calls.

**Tier 2 — Async agent runs.** Any agent run flagged "non-interactive" (already a property on `AgentGraphExecution` via the run-mode field) gets a batch eligibility check. If the run is single-shot LLM-heavy (no tool loop deeper than N steps, no human-in-the-loop blocks), route it through batch. This needs no UI change — just a route in the executor that picks batch over OpenRouter when eligibility passes.

**Tier 3 — Opt-in slow chat.** Add a `mode = "batch"` selector to the chat UI ("Slow mode, half price, response within 1 hour"). Frontend doesn't stream; instead, the session row carries a `batch_id` and `batch_state`, and the UI polls or subscribes to a notification when the result lands. This is the largest UX change and should ship last after Tiers 1–2 prove the plumbing.

## 5. Architecture sketch

```
┌──────────────┐    ┌────────────────────┐    ┌──────────────────────┐
│  Scheduler   │    │ copilot_executor   │    │  copilot_batch_exec  │
│  (dream job) │    │ (sync chat turns)  │    │  (new service)       │
└──────┬───────┘    └────────┬───────────┘    └──────────┬───────────┘
       │                     │                           │
       │   ────  RabbitMQ: copilot_batch_pending ───▶    │
       │                                                 │
       │                                          ┌──────┴──────┐
       │                                          │ accumulator │  every 5m
       │                                          │  + flush    │  or N items
       │                                          └──────┬──────┘
       │                                                 │
       │                                                 ▼
       │                                     POST /v1/messages/batches
       │                                                 │
       │                                          ┌──────┴──────┐
       │                                          │  poll loop  │  every 30s
       │                                          │  (Redis-pinned │
       │                                          │   batch state)  │
       │                                          └──────┬──────┘
       │                                                 │
       │   ◀──  enqueue_episode / append_message  ───────┘
```

**State lives in three places:**
- **RabbitMQ** queue `copilot_batch_pending` — newly enqueued items awaiting accumulation
- **Redis** key `batch:active:{batch_id}` → `{status, custom_id_map, submitted_at, expires_at}` — survives container restarts, drives the poll loop
- **Postgres** new table `MessageBatch` (id, anthropic_batch_id, user_ids[], status, results_summary, created_at) — durable audit trail; survives Redis flush

**New `mode = "batch"` semantics on `CoPilotExecutionEntry`:**
- Skips Redis Stream session creation (no SSE)
- Skips RabbitMQ direct routing; instead inserts into `MessageBatchItem` Postgres rows
- Returns `batch_item_id` for caller tracking
- On batch completion, the batch executor invokes the same `_publish_completion` machinery that synchronous turns use, so downstream consumers (notification bus, ratification metrics) don't notice the difference

## 6. Polling design

Naive polling burns budget. Conservative defaults:

- **Outer loop interval**: 30s for the first 5 minutes, then exponential to 5 min, capped at 5 min. Most batches return in <1 hour; an empty inner loop costs ~12 retrieve calls/hr per batch.
- **Backpressure**: never have more than `N` batches in flight per workspace (Anthropic enforces concurrent-batch caps; treat 5 as the safe default and feature-flag higher).
- **Restart safety**: poll-loop state lives in Redis; on container start, hydrate active batches from `MessageBatch` rows where `status = in_progress`, resume polling.
- **24-hour timer**: schedule a one-shot `expires_at` job per batch; on fire, force-poll and treat any `expired` items as failures (which Anthropic does not bill for).
- **Webhooks-when-they-ship**: leave a `webhook_url` registration shim behind a feature flag; when Anthropic GAs webhooks, flip the flag and drop polling.

## 7. Cost optimizations to bake in

- **1-hour cache TTL** for batch system prompts and memory context — batches frequently span >5 minutes, so the default 5m cache misses. Set `cache_control = {"type": "ephemeral", "ttl": "1h"}` on the cacheable system blocks.
- **Coalescing window** — for dreams, prefer fewer larger batches over many small ones. One batch with 500 users' dream requests cache-hits the shared instructions far better than 500 single-request batches.
- **Per-request `custom_id` = `{user_id}:{turn_id}`** for direct routing of results back to the originating session.
- **`max_tokens` ceiling per dream phase** to bound output cost; phase 1 (consolidate) ≤ 4k, phase 2 (recombine) ≤ 8k.

## 8. Failure modes

- **`expired` results**: 24h cap hit; user gets no dream that night. Mitigation: re-enqueue once with high priority; if a second batch also expires, drop and log.
- **Per-request `errored`**: bad params or model error; not billed. Surface to Sentry with `custom_id`; do not retry automatically (param errors won't fix themselves).
- **OAuth/API key rotation mid-batch**: batches use the key from submission time; key rotation is fine.
- **Cancellation race**: user deletes account while their batch is in flight. Cancel batch, but expect Anthropic to bill for already-in-flight items.
- **Pricing drift**: 50% discount is policy, not contract. Wire the cost calculator to read from a config table, not a constant.
- **Tool calls in batch**: batches support tool use, but tool *execution* happens in our process after the batch returns the tool-call response — meaning a batch with tool calls becomes multi-batch (each tool round-trip = one batch submission). For Tier 1 (dreams) we restrict tools to memory ops that we can simulate or defer; for Tier 2+ we need a "submit → execute tools locally → resubmit" loop with its own state machine. **Easiest v1: forbid tool calls in batch mode**; dream prompt is single-turn produce-this-JSON.

## 9. ZDR caveat

Batches are not ZDR-eligible. If we have or will have enterprise tenants on ZDR contracts, the batch path must be **opt-out by tenant flag**, not opt-in. Coordinate with legal/compliance before launch. For consumer users (the default), no change.

## 10. Build plan

In order. Each step is independently shippable behind a feature flag.

1. **`copilot/batch/client.py`** — thin wrapper over `anthropic.AsyncAnthropic().messages.batches`, with retry, rate-limit handling, and structured logging. ~150 LOC.
2. **Postgres schema** — `MessageBatch` + `MessageBatchItem` tables; Prisma migration. Add `batch_item_id` to `ChatMessage` (nullable).
3. **`copilot/batch/accumulator.py`** — consumes `copilot_batch_pending`, flushes on size or time. Redis-backed buffer. ~200 LOC.
4. **`copilot/batch/poller.py`** — restartable poll loop with Redis state. ~250 LOC.
5. **`copilot/batch/results.py`** — JSONL stream parser; routes results to the same handlers synchronous turns use. ~150 LOC.
6. **`copilot_batch_executor` service** — new container in `docker-compose.platform.yml` mirroring `copilot_executor`. Same image, different entrypoint.
7. **Wire dreams** — scheduler enqueues to `copilot_batch_pending` with `mode="batch"`. No new code in the scheduler beyond the existing `_execute_dream_pass()` from the scheduler plan.
8. **Tier 2: async agent runs** — eligibility check in `executor/manager.py` routes single-shot non-streaming runs through batch.
9. **Tier 3: opt-in slow chat** — `mode = "batch"` on the chat UI, new `BatchSession` API, frontend polling/notification.
10. **Webhook shim** — when Anthropic GAs batch webhooks, replace poll loop with webhook + final reconcile poll.

## 11. Open questions for product

1. Do we surface the 50% savings to users (consumer-facing "slow mode" pricing), or absorb it as margin?
2. Are there tenants on ZDR contracts today? If so, batch mode is opt-out for them.
3. Tier 2 — what's the eligibility heuristic for "non-interactive agent run"? Conservative: any run flagged `mode = "scheduled"`. Aggressive: any run whose graph has no human-input blocks and a measured P95 < 60s on previous executions.
4. SLA expectations — Anthropic's 24h hard cap is real. If we promise "within 1 hour" to users in Tier 3 UI, we need an internal fallback that re-routes to sync after, say, 45 minutes of no completion.
5. Cost reconciliation — Anthropic bills batches separately on the invoice. Make sure FinOps can attribute batch costs back to user/feature.

**Recommendation: ship Tier 1 (dreams) first**, behind `BATCH_PROCESSING_DREAMS` LaunchDarkly flag, before standing up Tier 2/3. The dream-pass scoping doc already assumed batch was coming; this just makes it concrete.
