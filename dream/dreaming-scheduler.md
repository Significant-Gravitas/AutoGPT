# Dream-Pass Scheduler: Technical Investigation

## 1. Per-User Timezone Handling

**Storage.** The timezone is a first-class column on the Postgres `User` model, not in a profile sidecar.

- `schema.prisma:41` — `timezone String @default("not-set")`
- `backend/data/model.py:50` — sentinel constant `USER_TIMEZONE_NOT_SET = "not-set"`
- `backend/data/model.py:103-107` — `User.timezone: str = Field(default=USER_TIMEZONE_NOT_SET, description="User timezone (IANA timezone identifier or 'not-set')")`

**Write path.** `update_user_timezone()` at `data/user.py:502-517` performs a Prisma `.update()` and invalidates the in-process LRU cache. The API surface is `PUT /auth/user/timezone`, validated by `TimeZoneName` (`util/timezone_name.py`), a custom `str` subclass backed by `pytz.all_timezones`. Only canonical IANA names (`America/New_York`, `Europe/London`, etc.) pass — numeric offsets like `+05:30` are rejected at validation time.

**How `add_graph_execution_schedule()` gets the timezone.** The parameter is `user_timezone: str | None = None` at `scheduler.py:669`. At `scheduler.py:683-690`, if the value is falsy the scheduler emits a warning and sets `user_timezone = "UTC"`. The API layer at `v1.py:1741-1756` reads it from the request body if present; otherwise it calls `get_user_timezone_or_utc(user.timezone)` from `util/timezone_utils.py:133-150`, which maps `"not-set"` and `None` to `"UTC"`. The resolved string is passed into `CronTrigger.from_crontab(cron, timezone=user_timezone)` at `scheduler.py:711` — APScheduler accepts any IANA name that `ZoneInfo` can resolve.

**Dream-pass implication.** The dream-pass job must call `get_user_timezone_or_utc(user.timezone)` before registering. Users who have never set a timezone will dream at 3am UTC rather than 3am local — acceptable for v1 but worth surfacing in copy. There is no fallback to browser-detected timezone server-side.

---

## 2. Job-ID Collisions Across Worktrees / Dev Envs

**Job store URL.** At `scheduler.py:479` the scheduler reads `DIRECT_URL` from the environment:

```python
db_schema, db_url = _extract_schema_from_url(os.getenv("DIRECT_URL"))
```

`_extract_schema_from_url()` at `scheduler.py:64-80` pulls the `?schema=` query param (defaults to `"public"`, but `.env.default:15` sets `DB_SCHEMA=platform`) and strips it from the URL. Jobs land in `{schema}.apscheduler_jobs`. In the standard Docker Compose stack (`docker-compose.platform.yml:27-28`) every service uses the same hardcoded connection string pointing at the `db` container with schema `platform`. There is no env-name or worktree namespace anywhere in the table name or job ID.

**Safety analysis for `dream_pass_{user_id}`.** Within a single DB instance, user IDs are Supabase UUIDs — globally unique. `dream_pass_883cc9da-...` cannot collide with any other user's job. Across environments sharing the same DB (not the recommended topology, but possible for staging), two deployments writing `dream_pass_{same_user_id}` would stomp each other via `replace_existing=True` — the later write wins, resetting the next fire time. This is the same risk that exists for graph execution schedules today and is acceptable if environments use separate DB instances (which is the current practice for dev/staging/prod).

**Three separate job stores.** System jobs go into `EXECUTION` (`apscheduler_jobs`); batched notification jobs go into `BATCHED_NOTIFICATIONS` (`apscheduler_jobs_batched_notifications`); weekly notifications use in-memory `WEEKLY_NOTIFICATIONS` (`scheduler.py:494-517`). Dream-pass jobs should use the `EXECUTION` store, matching existing user-triggered schedules.

---

## 3. Idempotency Under Failure

**APScheduler's global defaults** at `scheduler.py:489-492`:

```python
"coalesce": True,        # only run the latest missed fire-time, skip duplicates
"max_instances": 1000,   # effectively unlimited concurrency per job
"misfire_grace_time": None,  # no staleness cap on missed jobs
```

With `coalesce=True` and a nightly cron, if the scheduler container is killed and restarted within a single 3am window, APScheduler will run the job once on restart. If the scheduler is down for multiple nights, it runs once (coalesced) on restart. There is no auto-resume: a killed mid-pass execution is lost.

**No distributed lock.** APScheduler's `max_instances` counter is in-memory on the scheduler process. If a second scheduler pod runs (not current topology but possible with horizontal scaling), both pods can execute `dream_pass_{user_id}` concurrently. Two simultaneous Claude passes over the same user's episodes will produce duplicate or contradictory tentative memories.

**Recommendation: Redis SETNX lock.** At the start of `_execute_dream_pass()`, call `SETNX dream:inflight:{user_id} 1 EX 1800` (30-minute TTL). If the key already exists, log and return early. Clear the key in a `finally` block. This is idempotent under restarts (TTL auto-expires) and safe under multi-pod execution. The per-user `asyncio.Queue` in `ingest.py` already serializes Graphiti writes within a process; the Redis lock extends that guarantee across processes.

---

## 4. One-Job-at-a-Time Semantics

**`max_instances` is per job-id, not per function.** APScheduler stores an instance counter on the job object keyed by job id. The `ensure_embeddings_coverage` job demonstrates this at `scheduler.py:600-607`:

```python
self.scheduler.add_job(
    ensure_embeddings_coverage,
    id="ensure_embeddings_coverage",
    max_instances=1,   # scoped to this job id only
    ...
)
```

Because `dream_pass_{user_id}` produces a distinct job id per user, setting `max_instances=1` on each registration gives per-user isolation: `dream_pass_alice` and `dream_pass_bob` maintain independent counters, so Alice's dream does not block Bob's. Two calls to `add_dream_pass_schedule` for the same user with `replace_existing=True` are idempotent — the second call updates the trigger but does not create a second job.

**Multi-pod caveat.** As noted in §3, the instance counter is in-memory only. `max_instances=1` per job plus the Redis SETNX lock together cover both single-pod and multi-pod cases.

---

## 5. Triggering Off-Cycle

**The `@expose` + `endpoint_to_async` pattern.** Every `@expose`-decorated method on `Scheduler` is callable remotely over HTTP from any service via `SchedulerClient`. The pattern for on-demand triggers is to add a new `@expose` method that calls the job function directly, bypassing the cron schedule. This is already the convention for all system jobs:

- `scheduler.py:792-794`: `execute_ensure_embeddings_coverage(self)` calls `ensure_embeddings_coverage()` synchronously.
- `scheduler.py:769-770`: `execute_report_late_executions(self)` calls `report_late_executions()` directly.

**There is no `scheduler.modify_job()` or `scheduler.run_job_now()` usage anywhere in the codebase.** APScheduler does expose `job.modify(next_run_time=datetime.now(tz=utc))` to force an immediate fire, but this goes through the thread pool and respects `max_instances`, making it less suitable than a direct call. The direct-call pattern via `@expose` matches conventions, is simpler, and allows the body to be separately unit-tested.

**Dream-pass on-demand interface:**

```python
@expose
def execute_dream_pass(self, user_id: str) -> dict:
    """Manually trigger a dream pass for a specific user."""
    return run_async(_execute_dream_pass(user_id))
```

Wire to `SchedulerClient`:

```python
execute_dream_pass = endpoint_to_async(Scheduler.execute_dream_pass)
```

This is callable from an admin API endpoint, a post-conversation hook, or a Managed Agents Dreams webhook with no changes to the scheduler infrastructure.

---

## 6. Resource Budget

**Thread pool sizing.** The `BackgroundScheduler` thread pool at `scheduler.py:484-487`:

```python
"default": ThreadPoolExecutor(
    max_workers=self.db_pool_size()
),
```

`db_pool_size()` returns `Config().scheduler_db_pool_size`, which defaults to `3` at `settings.py:288-291`. At most 3 jobs run concurrently on the scheduler. With 10,000 users whose local 3am spans 24 time zones, the burst arrival rate is roughly 417 dream passes per hour, or 7 per minute. A dream pass that blocks a thread for 2-10 minutes (Claude call over 100 sessions) would saturate the pool almost instantly, starving late-execution alerts, OAuth cleanup, and embedding backfill.

**The scheduler is the wrong place to run the dream body.** The `run_async()` wrapper at `scheduler.py:133-141` blocks the calling thread until the coroutine resolves. A 5-minute async coroutine holds a thread for 5 minutes. Three such coroutines fill the pool.

**Correct architecture: the scheduler enqueues, the copilot executor runs.** Publish a `dream_pass_requested:{user_id}` message to RabbitMQ from the scheduler job. The CoPilot executor (already consuming RabbitMQ queues, already async, already horizontally scalable) picks it up and runs the Claude call. This matches the existing chat turn pattern: `routes.py` enqueues to RabbitMQ; the copilot executor dequeues and processes. The scheduler thread is freed in milliseconds after publishing the message. No thread pool changes required.

---

## 7. Observability

**Existing event listeners** at `scheduler.py:622-624`:
- `job_listener` — logs `"Job {id} completed successfully."` or `"Job {id} failed: {exc type}: {exc}"` (unstructured, no user_id).
- `job_missed_listener` — logs when a cron fire-time was missed.
- `job_max_instances_listener` — logs when the instance limit is hit.

**Prometheus.** `SCHEDULER_JOBS = Gauge("autogpt_scheduler_jobs", ..., labelnames=["job_type", "status"])` is declared at `instrumentation.py:51-55` but has no `observe()` or `.set()` call anywhere in the codebase — it is a stub. Adding dream-pass metrics requires new instruments:

```python
DREAM_PASS_DURATION = Histogram(
    "autogpt_dream_pass_duration_seconds", ...,
    buckets=[30, 60, 120, 300, 600, 1200],
)
DREAM_PASS_TOKENS = Counter(
    "autogpt_dream_pass_tokens_total", ..., labelnames=["direction"]
)
DREAM_PASS_MEMORIES = Counter(
    "autogpt_dream_pass_memories_total", ..., labelnames=["outcome"]
    # outcome: proposed | ratified | superseded | dropped
)
```

**Structured logging.** The existing listeners use unformatted f-strings with no structured fields. Dream-pass logging should emit `user_id`, `duration_ms`, `memories_proposed`, `memories_written` as structured extra fields so log queries work without regex.

**User-facing telemetry.** The four metrics the scoping doc recommends (warm-context relevance score, contradiction-detection rate, ratification rate, cache-hit rate) are not infrastructure — they require post-dream query instrumentation in the Graphiti read path and the warm-context assembly in `service.py`. Plan these as separate instrumentation tasks.

---

## 8. Tests

**Current state.** `scheduler_test.py:11-41` is the only scheduler test. It uses `SpinTestServer` (live Postgres, live APScheduler with `register_system_tasks=False`), asserts CRUD on graph execution schedules over the RPC client, and does not advance the clock or wait for a job to fire. There is no fake-tick API in APScheduler 3.x.

**Fake-tick options ranked by weight:**
1. **Unit-test `_execute_dream_pass()` directly** — mock `get_graphiti_client`, `prisma`, and the Claude SDK; call the coroutine with `pytest.mark.asyncio`. No scheduler, no Postgres, no SpinTestServer. This covers 90% of the interesting logic.
2. **Integration test via `SpinTestServer` + `execute_dream_pass()`** — register the schedule, call the on-demand trigger (which bypasses the cron timer), assert on mocked Graphiti calls. Mirrors `test_agent_schedule`.
3. **Clock-advance integration** — set `next_run_time` via APScheduler's internal API and call `scheduler._scheduler.wakeup()`. Fragile; avoid.

**Lightest pattern for "schedule → ran → wrote memories":**

```python
@pytest.mark.asyncio(loop_scope="session")
async def test_dream_pass_lifecycle(server: SpinTestServer):
    await db.connect()
    user = await create_test_user()
    scheduler = get_scheduler_client()

    # Register
    job = await scheduler.add_dream_pass_schedule(
        user_id=user.id, user_timezone="UTC"
    )
    assert job.id == f"dream_pass_{user.id}"

    # On-demand trigger with mocked body
    with patch("backend.copilot.dream.pass._execute_dream_pass") as mock_dream:
        mock_dream.return_value = {"memories_proposed": 3, "memories_written": 2}
        result = await scheduler.execute_dream_pass(user.id)

    assert result["memories_proposed"] == 3

    # Tear down
    await scheduler.delete_dream_pass_schedule(user.id)
    jobs = await scheduler.get_dream_pass_schedules()
    assert not any(j.id == f"dream_pass_{user.id}" for j in jobs)
```

---

## Scheduler-Side Build Plan

Files to touch, in dependency order:

1. **`backend/copilot/dream/pass.py`** (new) — `async def _execute_dream_pass(user_id: str) -> dict`. Session fetcher, Graphiti episode fetcher, Claude call, `enqueue_episode()` writes, metrics, Redis lock acquire/release. No APScheduler imports.

2. **`backend/executor/scheduler.py`** — import `_execute_dream_pass`; add:
   - `def execute_dream_pass_sync(user_id: str) -> dict: return run_async(_execute_dream_pass(user_id))` — thread-pool entry point.
   - `@expose def add_dream_pass_schedule(self, user_id, user_timezone)` — `CronTrigger.from_crontab("0 3 * * *", timezone=user_timezone)`, id=`dream_pass_{user_id}`, `max_instances=1`, jobstore=`EXECUTION`.
   - `@expose def delete_dream_pass_schedule(self, user_id)` — removes `dream_pass_{user_id}`.
   - `@expose def execute_dream_pass(self, user_id) -> dict` — direct on-demand trigger.
   - Wire all three into `SchedulerClient` with `endpoint_to_async`.

3. **`backend/data/model.py`** — add `last_dream_at: datetime | None = None`, `dream_enabled: bool = True`, `dream_config: dict = Field(default_factory=dict)` to `ChatSessionMetadata`. JSON column, no migration.

4. **`backend/monitoring/instrumentation.py`** — add `DREAM_PASS_DURATION`, `DREAM_PASS_TOKENS`, `DREAM_PASS_MEMORIES` Prometheus instruments. Wire the existing `SCHEDULER_JOBS` gauge.

5. **`backend/copilot/dream/pass_test.py`** (new) — unit tests for `_execute_dream_pass` with all external calls mocked. No `SpinTestServer`.

6. **`backend/executor/scheduler_test.py`** — add `test_dream_pass_lifecycle` following the pattern above.

7. **`backend/api/features/dream/routes.py`** (new) or extend `v1.py` — admin endpoint `POST /admin/users/{user_id}/dream-pass` calling `scheduler.execute_dream_pass(user_id)`, guarded by `requires_admin_user`.

8. **`backend/util/settings.py`** — add `dream_pass_enabled: bool = Field(default=False)` as a feature-flag gate so the job is only registered for users where it is enabled. Avoids scheduling 10k jobs on day one.

---

## Essential Files for Understanding This Topic

- `autogpt_platform/backend/backend/executor/scheduler.py`
- `autogpt_platform/backend/schema.prisma` (lines 17-84)
- `autogpt_platform/backend/backend/data/model.py` (lines 50-107)
- `autogpt_platform/backend/backend/data/user.py` (lines 502-517)
- `autogpt_platform/backend/backend/util/timezone_utils.py`
- `autogpt_platform/backend/backend/util/timezone_name.py`
- `autogpt_platform/backend/backend/copilot/graphiti/ingest.py`
- `autogpt_platform/backend/backend/copilot/graphiti/memory_model.py`
- `autogpt_platform/backend/backend/copilot/model.py` (lines 48-56)
- `autogpt_platform/backend/backend/executor/scheduler_test.py`
- `autogpt_platform/backend/backend/util/test.py` (lines 29-74)
- `autogpt_platform/backend/backend/monitoring/instrumentation.py` (lines 51-55)
- `autogpt_platform/backend/backend/util/settings.py` (lines 288-291)
- `autogpt_platform/backend/backend/util/service.py` (lines 838-849)
- `autogpt_platform/backend/backend/api/features/v1.py` (lines 1741-1756)
