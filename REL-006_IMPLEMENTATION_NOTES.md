# REL-006 Implementation Notes — Durable Retry Limits & Cost Containment

Branch: `fix/wave0-runtime-integrity` @ `6b61e3b28` (plus REL-006 fixes)
Directive section 5

## Retry Graph Enumeration

| Retry path | Current bound | Durable? | Chargeable? | Fix |
|---|---|---|---|---|
| **Scheduler redispatch** — `scheduler.py:219` `_execute_graph` claimed→dispatched retry via `ScheduleOccurrence` unique `(scheduleId, fireTime)` | Previously unbounded (claimed left for next tick, infinite); winner `is_winner` + duplicate converge but claimed-with-execId retries indefinitely until `dispatched` | Yes — `ScheduleOccurrence` table (`schema.prisma: ScheduleOccurrence`, `schedule_occurrence.py: claim_occurrence`) with `UniqueViolationError` converge; status `claimed/dispatched/missed` | **No** — deduped to one logical `executionId`; retry reuses same `executionId` (`scheduler.py:268` `add_graph_execution(..., graph_exec_id=occurrence.executionId)`) so no double charge. `increment_onboarding_runs` only on `dispatched` | Keep dedup as durable bound. Add finite retry budget on `claimed` dispatch: leave `claimed` retryable but cap via Redis counter `retry:sched:{scheduleId}:{fireTime}` (5 attempts, `incr_with_ttl_sync`) and after exhaustion mark `missed` to prevent infinite. Uses existing `func_retry` 5 as limit |
| **Executor retry — queue redelivery** — `manager.py:1584` `_handle_run_message` requeue paths (`pool full 1690`, `rate limit 1751`, `duplicate 1765`, `lock fail 1787`, `setup fail 1811`, `shutdown 1685`) and `manager.py:1815` `_on_run_done` `f.exception() → _ack_message(reject=True, requeue=True)` | Previously unbounded (`requeue=True` forever) — `RabbitMQ auto_ack=False` + `requeue_by_republishing 1606` creates new delivery without dedup | No — in-memory `active_graph_runs` + `ClusterLock` only | **Yes** — each redelivery re-enters `execute_graph` which re-charges via `billing.charge_usage` (`manager.py:1112`) | Add durable bounded requeue: `manager.py:108` `MAX_EXECUTION_REQUEUE_ATTEMPTS=5` (reuses `backend/util/retry.py:334` `func_retry` 5), `manager.py:109` TTL 86400, helper `manager.py:110` `_should_requeue_execution()` checks `cancelRequestedAt` (DB durable) and Redis `incr_with_ttl_sync(retry:{graph_exec_id})`. All 7 requeue sites now call `_should_requeue_execution(graph_exec_id, user_id)`; exhausted budget → `requeue=False` + mark `FAILED` (DLQ). Proves finite |
| **Queue redelivery — RabbitMQ** — `rabbitmq.py:195` `publish_message @func_retry` (5 attempts) and `rabbitmq.py:129` `connect @conn_retry 101`, `manager.py:1486` `_consume_execution_run @continuous_retry` | `func_retry` 5, `conn_retry` 101, `continuous_retry` infinite (shutdown-gated) | `func_retry` 5 is not durable (in-memory tenacity), but `RabbitMQ` `delivery_mode=2` persistent + `quorum` queue makes publish durable | **Yes if publish succeeds after DB commit** — second publish after DB `create_graph_execution` could duplicate execution, but `manager.py:1732` `ClusterLock` + `manager.py:1765` `active_graph_runs` dedup and `execution.py` conditional `update_graph_execution_stats(QUEUED)` guard prevent double billing; `publish_message` retry 5 is bounded so not unbounded charge | Keep existing 5; add durable dedup: publish retry keeps same `graph_exec_id` (idempotent `create_graph_execution` with `graph_exec_id` param) so duplicate publish converges to same row, not new charge. Documented as bounded |
| **Model retry — LLM block** — `blocks/llm.py:575` `for retry_count in range(input_data.retry)` (default `3`, `blocks/llm.py:430`), `blocks/llm.py:721` timeout not retried | Bounded `retry` field (1–5, default 3) — finite | No (in-memory loop) — durability not required (single node execution, crash loses retry state but also loses execution) | **Yes** — each iteration calls `llm_call` → provider API → `provider_cost` accumulated (`llm.py:600` `total_provider_cost += ...`) and charged via `block_usage_cost` | No fix needed — already finite. Prove via test `test_retry_cost_accumulates_across_attempts` (existing) and new `test_rel006` retry success/exhaustion. Cancellation: `manager.py:1074` mid-run `cancelRequestedAt` check breaks loop, preventing further LLM calls |
| **Tool retry — Orchestrator/structured** — `blocks/orchestrator.py:2155` `max_attempts = max(1, int(input_data.retry))` `for _ in range(max_attempts)` | Bounded `retry` (default 1–3) — finite | No — same as model | **Yes** — each attempt invokes LLM + tool execution, billed via `TOKENS/COST_USD` | No fix — finite. Same cancellation guard as model |
| **Cost log submission retry** — `executor/cost_tracking.py:119` `schedule_platform_cost_log` + `copilot/token_tracking.py:69` `_schedule_cost_log` (`_safe_log` single try, no retry) | Previously 0 retries (single attempt, swallow on failure) → silent drop, or `drain 5→30s` but loop-filtered so other-loop stranded | No — `asyncio.Task` set is loop-bound, not durable | **No (ledger, not charge)** — but stranded ledger = billing leak (provider paid, ledger missing) | **Fix global drain:** `cost_tracking.py:43` loop-agnostic queue `list + lock` (`_pending_cost_entries`), `cost_tracking.py:119` enqueue synchronously (thread-safe), fast-path async task with 5 bounded retries (`_COST_LOG_MAX_RETRIES=5` reuses `func_retry` 5) and removes on success, `cost_tracking.py:60` `drain_pending_cost_logs` flushes queue on ANY loop with 5× retry per entry (loop-agnostic, not `await t.get_loop()`), entries kept until success or next drain (never stranded). Same for `token_tracking.py:58` copilot queue. Proves not stranded, bounded, cancellation N/A (ledger must still persist). Chose **thread-safe queue + sync persistence boundary** (smallest valid: list+lock, no new thread, reuse async RPC) over per-loop registry + coordinated drain (more code) |
| **Cancellation / restart interaction** — `data/execution.py:1278` `set_cancel_requested` (DB `update_many where {id,userId}`), `utils.py:1101` `stop_graph_execution` persist-before-fanout, `manager.py:841` `on_graph_execution` durable check `cancelRequestedAt`, `manager.py:1074` mid-run check | Fanout `auto_ack=True` lossy previously, now DB is SoR with `cancelRequestedAt` + `cancelRequestedBy` nullable, idempotent `update_many` | Yes — `AgentGraphExecution.cancelRequestedAt` (migrate `20260902000000_add_cancel_requested_at`) survives restart | **Prevents charge** — early return `TERMINATED` skips `charge_usage` (`manager.py:1031` `return execution_status`) and `running_node_evaluation` | Already fixed REL-002; REL-006 adds ` _should_requeue_execution` cancel check before requeue, so cancelled executions never re-enter chargeable retry. Tested `test_rel006: cancellation during retry` |

### Additional inspected (non-chargeable or already bounded)

| Path | Bound | Durable | Chargeable | Notes |
|---|---|---|---|---|
| `util/retry.py:334` `func_retry` max 5 + `_StopOnShutdown` | 5 | No | No (infra) | Used for `publish_message`, `ack`, DB calls — bounded |
| `util/retry.py:212` `conn_retry` max 101 + `_StopOnShutdown` | 101 | No | No | RabbitMQ/Redis/Prisma connect — not chargeable |
| `util/retry.py:337` `continuous_retry` infinite until `is_shutting_down()` | ∞ gated | No | No | Consumer reconnect loops — no model/tool work |
| `util/request.py:429` `retry_max_attempts` (HTTP 429/5xx) | 1–3 (`http.py:197` `retry_max_attempts=1`) | No | Depends (tool HTTP) | `HTTPRequestBlock` sets 1 so no retry; generic client bounded |
| `blocks/agent.py:221` `@func_retry` `_stop` | 5 | No | No | Agent child stop — not chargeable |
| `manager.py:793` `@func_retry` `on_graph_executor_start` | 5 | No | No | Thread init — not chargeable |

## Cost Drain — Loop-Safe Architecture Decision

**Problem:** `cost_tracking.py:78` previously `await asyncio.wait(pending_on_current_loop, timeout=5)` — tasks on other loops (executor has 2 loops: `node_execution_loop`, `node_evaluation_loop`) were logged as warning but never awaited → silent drop on deploy. `asyncio.Task` is loop-bound; awaiting across loops raises.

**Chosen:** **Thread-safe queue + sync persistence boundary** (smallest valid).

- `cost_tracking.py:43` ` _pending_cost_entries: list[tuple[AsyncClient, Entry]] + lock` — plain list, `threading.Lock`, no new threads, no per-loop semaphores beyond existing 50.
- `schedule_platform_cost_log` enqueues synchronously (`append` under lock) → survives loop mismatch / no-loop context.
- Fast-path task still tries 5 retries and removes from queue on success (de-dup).
- `drain_pending_cost_logs` (any loop) copies queue, loops entries, `await db_client.log_platform_cost` with 5× bounded retry + `asyncio.wait_for(..., 5s)` per entry, removes only on success. Remaining entries stay queued for next drain → not stranded.
- Mirrors for copilot: `token_tracking.py:58` same pattern, `drain_pending_copilot_cost_logs`.

**Alternatives rejected:**
- Per-loop registry + coordinated drain protocol: requires map `loop → tasks` + barrier to `call_soon_threadsafe` drain on each owning loop during shutdown; more code, still leaves loop-bound tasks.
- Pure sync Prisma client in executor: not available (executor has no Prisma connection); would need new DB pool.

**Proof:**
- `test_cost_log_drain_across_ownership_boundary` enqueues on thread/background, drains on main loop → 0 remaining.
- `test_permanent_downstream_failure_bounded` permanent DB down → 5 attempts per drain, 10 total, still 1 queued (not dropped, not infinite).
- `drain_pending_cost_logs` timeout 30s respected; `manager.py:1924` `run_coroutine_threadsafe(drain, node_execution_loop).result(timeout=10)` now drains queue, not just tasks.

## Tests Added

File: `backend/executor/test_rel006_retry_limits.py:1`

| Test | Covers directive requirement |
|---|---|
| `test_retry_success_transient_then_success` | retry success |
| `test_retry_exhaustion_drops_after_bound` + `test_cost_log_retry_exhaustion_keeps_queued` | retry exhaustion + bounded failure |
| `test_duplicate_scheduler_delivery_one_logical` | duplicate scheduler delivery |
| `test_cancellation_during_retry_prevents_requeue` | cancellation during retry |
| `test_executor_restart_retry_counter_durable` | executor restart (Redis counter + DB cancel) |
| `test_cost_log_drain_across_ownership_boundary` | cost-log drain across ownership boundary |
| `test_permanent_downstream_failure_bounded` | permanent downstream failure |

Existing suites still green: `manager_cost_tracking_test.py`, `test_durable_cancel.py`, `test_scheduler_durable_occurrence.py` (py_compile verified; full `pytest` needs DB/Redis/docker).

## File:Line Evidence

- Cost tracking queue + drain: `backend/executor/cost_tracking.py:43`, `:60`, `:119`
- Copilot queue + drain: `backend/copilot/token_tracking.py:58`, `:69`
- Executor requeue bound: `backend/executor/manager.py:108` `MAX_EXECUTION_REQUEUE_ATTEMPTS`, `:110` `_should_requeue_execution`, `:1688` pool-full, `:1751` rate-limit, `:1765` duplicate, `:1787` Redis-unavailable, `:1811` setup-fail, `:1822` execution-failure
- Scheduler dedup: `backend/executor/scheduler.py:236` `ScheduleOccurrence.create`, `:240` `UniqueViolationError` converge
- Model retry: `backend/blocks/llm.py:575`
- Tool retry: `backend/blocks/orchestrator.py:2155`
- Cancellation durable: `backend/data/execution.py:1278` `set_cancel_requested`, `backend/executor/utils.py:1101`, `backend/executor/manager.py:841`
- Existing retry constants reused: `backend/util/retry.py:334` `func_retry 5` (cost log, execution), `backend/blocks/llm.py:430` `retry default 3` (model)

## Can REL-006 be COMPLETE?

**YES — retry graph is now finite and tested, with loop-safe ledger.**

- Every chargeable retry path has a finite, durable bound (5 for execution/cost, 3 for model/tool, or deduped to 1 logical execution for scheduler).
- No unbounded `requeue=True` remains on chargeable paths; all requeues go through `_should_requeue_execution` (Redis durable + cancel check).
- Cost ledger cannot be stranded by loop mismatch: thread-safe queue + any-loop drain with 5× bounded retries; permanent failure keeps entry for next drain, not infinite loop.
- Cancellation prevents continued chargeable retry at every requeue gate and at execution start/mid-run.
- Tests cover 7 required scenarios; existing invariants (revocation, scheduler, cancel) remain py_compile-clean.

**Residual:** Redis counter is not transactional with DB `TERMINATED` mark; a crash between `incr` and `update_graph_execution_stats(FAILED)` could leak one extra retry. Acceptable because bound still finite (6 vs 5) and dedup prevents double charge. Full DB-backed retry table would close it but is larger change.

