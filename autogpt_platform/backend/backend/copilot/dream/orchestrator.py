"""Three-phase dream-pass orchestrator (sync_baseline path).

Walks a user's recent memory window through the consolidate → recombine
→ sanitize pipeline, then applies the sanitizer's ``DreamOperations``
(clamped by ``clamp.py``) to Graphiti + Postgres.

Each phase here is one ``inference.complete.structured_complete`` call
on the chat transport's provider, traced and recorded through the
inference package. When ``routing.resolve_dream_execution_path``
picks ``anthropic_batch``, ``batch_handoff.submit_dream_pass_batch``
submits the first phase to Anthropic's Message Batches API instead and
``batch_callbacks`` runs the later phases and the apply step as the
results land.

Every pass, on either route, gets a durable ``DreamPass`` row
(``store.py``): inserted with its lease at the start, advanced after each
step, and closed with how the pass ended, a write attempted before the pass
releases its lock. Like every record write it is best-effort: one that fails
leaves the row open behind a free lock until a later pass's guard or the
reaper (``reaper.py``) closes it. A pass holding the lock runs ``guard.py``
first, then before each phase and apply checks for a stop (``cancel.py``) and
renews its lease (``lease.py``); the batch route checks before its submit.

The orchestrator never raises out — every failure becomes a
``DreamPassResult`` with ``error`` set and, on the sync route, the usage
of every phase billed before it (a phase whose charge failed included),
so the admin trigger always gets a structured response back.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TypeVar

from pydantic import BaseModel, ConfigDict

from backend.copilot.config import ChatConfig
from backend.copilot.graphiti.scope import MemoryScope
from backend.copilot.inference.complete import structured_complete
from backend.copilot.inference.context import (
    InferenceContext,
    InferenceError,
    InferenceScope,
)
from backend.copilot.inference.routing import resolve_route
from backend.copilot.inference.trace import trace
from backend.copilot.trial_cost_context import attributed_usage
from backend.util.feature_flag import Flag, is_feature_enabled

from .apply import apply_operations, drain_status_from_stats
from .batch_handoff import submit_dream_pass_batch
from .billing import PhaseChargeError, check_dream_budget, record_phase_cost
from .cancel import stopped_error
from .citations import fact_scopes
from .clamp import clamp_pass_operations
from .fetch import (
    DreamInput,
    EpisodeRow,
    gather_dream_input,
    is_dream_authored_episode,
    parse_episode_timestamp,
)
from .guard import GUARD_BUDGET_SECONDS, guard_dream_pass
from .lease import admit_sync_apply, checkpoint
from .locks import (
    LOCK_CHECK_TIMEOUT_SECONDS,
    DreamLockHandle,
    DreamLockHeld,
    DreamLockLostError,
    dream_lock,
)
from .pass_errors import pass_error
from .pass_record import DreamTrigger
from .pass_run import DreamPassRun, PassEnded
from .phase_jobs import phase_job
from .prompts import (
    build_consolidate_prompt,
    build_recombine_prompt,
    build_sanitize_prompt,
)
from .routing import ExecutionPath, resolve_dream_execution_path
from .schemas import (
    ConsolidationOutput,
    DreamOperations,
    DreamOperationsSnapshot,
    DreamPassResult,
    DreamPhase,
    IngestionDrainStatus,
    PhaseUsage,
    RecombinationOutput,
)
from .store import (
    RECORD_WRITE_TIMEOUT_SECONDS,
    record_applying,
    record_gathered,
    record_phase_output,
    record_sync_outcome,
    start_pass,
)
from .usage import phase_usage

logger = logging.getLogger(__name__)


# Per-step temperature + max output token budgets. Spec ref:
# ``dream/p0-spec.md`` §2. Named by what the step does rather than its
# position; the consolidation→recombination→sanitization order is
# enforced by the call graph below.
CONSOLIDATE_TEMP = 0.2
RECOMBINE_TEMP = 0.9
SANITIZE_TEMP = 0.0
# The budget covers thinking as well as the JSON, and the standard slot can
# run a model that thinks unasked. Sonnet 5 through OpenRouter spent all of
# 4096 on reasoning over a 31-fact, 31-episode window (a 22.5K-token prompt)
# and answered nothing, failing every pass of that account at this phase;
# given 16384, the same call answered in 5,912 tokens, 5,068 of them
# reasoning. Budget a model does not spend costs nothing.
CONSOLIDATE_MAX_TOKENS = 16384
# Recombine + sanitize emit list-heavy JSON (up to 30 writes + 20
# proposals + 10 demotions, each with uuid arrays). At 8192 the
# response truncates mid-array and the JSON-balanced-brace extractor
# returns None, failing the whole phase. 16384 covers worst-case +
# headroom; cost impact is bounded by the per-phase caps anyway.
RECOMBINE_MAX_TOKENS = 16384
SANITIZE_MAX_TOKENS = 16384

# Per-phase LLM wall-clock ceilings, passed on each phase's inference job
# → call_provider. These exist because a single shared default can't fit
# every phase: recombine/sanitize carry 16384-token output budgets precisely
# because real responses exceed 8192 tokens, and at real decode speeds
# (Opus-class ~40-60 tok/s via OpenRouter) an 8-16K response takes 150-400s,
# so a ceiling under that truncates exactly the responses the token-cap
# budgets above exist to produce.
#
# A conservative ~20 tok/s decode floor would ask for max_output_tokens/20
# ≈ 205s / 819s / 819s per phase, but that sums to 1843s of LLM budget
# alone — past both SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS (1800s,
# scheduler.py) and DEFAULT_LOCK_TTL_SECONDS (1800s, locks.py) before
# fetch/apply even run. So the phases get values that fit the envelope
# instead. Budget math (three explicit line items, with real slack):
#
#   consolidate 240s + recombine 600s + sanitize 480s   = 1320s LLM ceiling
#   + apply.INGESTION_DRAIN_TIMEOUT_SECONDS 300s          drain cap
#   + DREAM_NON_LLM_HEADROOM_SECONDS 120s (budget check + fetch + enqueue
#     + demotions + summary + cost logging, EXCLUDING the drain)
#   = 1740s <= 1800s SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS  (60s slack)
#
# The pass's own bounded round trips to its record, its guard and its lock
# come on top: DREAM_STORE_IO_SECONDS, 179s should every one of them run to
# its deadline (a stalled DatabaseManager or Redis). Then a pass whose phases
# also run to their ceilings outlasts the scheduler's wait. That wait is not
# a kill: ``run_async`` stops waiting and leaves the pass running, and its
# lease renewals keep its lock and its row's lease fresh, so neither a newer
# pass's guard nor the reaper takes it for dead, and it finishes.
#
# The drain no longer counts against the lock TTL: apply renews the dream
# lock to a fresh budget right before the drain (see
# apply.LOCK_DRAIN_RENEWAL_SECONDS), so the lock only has to cover
# LLM + non-drain headroom + store I/O = 1619s <= 0.9 * DEFAULT_LOCK_TTL_SECONDS
# (1620s), even with every store round trip at its deadline (and a lease
# renewal before each phase shortens that stretch further), and the drain
# itself runs under the renewed lock. Both invariants are pinned by
# test_phase_timeouts_plus_headroom_fit_scheduler_and_lock_envelope —
# bumping any value here fails that test until the budget is re-balanced.
#
# Recombine (fast_advanced_model, Opus-class — the slowest decoder) gets
# the largest share: 600s covers 16384 tokens at ~27 tok/s. Sanitize runs
# on the faster fast_standard_model: 480s covers 16384 at ~34 tok/s.
# Consolidate's 240s covers its whole 16384 at ~68 tok/s, and a thinking
# answer the size measured above (5,912 tokens) at ~25 tok/s.
CONSOLIDATE_TIMEOUT_SECONDS = 240
RECOMBINE_TIMEOUT_SECONDS = 600
SANITIZE_TIMEOUT_SECONDS = 480
# Reserved for the non-LLM segments of the pass OTHER than the ingestion
# drain and the store round trips below (budget check, fetch, enqueue,
# demotions, summary write, cost logging). The drain is budgeted separately
# as apply.INGESTION_DRAIN_TIMEOUT_SECONDS and covered by a lock renewal.
DREAM_NON_LLM_HEADROOM_SECONDS = 120
# The sync pass's bounded round trips to its record and its lock, each at
# its own deadline. Record reads and writes: the RUNNING insert, the
# gathered window, before each phase a stop check, a lease write and the
# phase's output, the APPLYING write, apply's admission (a stop check and a
# lease write), and the outcome. Lock round trips: the lease extend before
# each phase, apply's admission (a lock read and an extend), and apply's own
# renewals, before its first write and before its drain. Plus the guard's
# whole budget.
_RECORD_ROUND_TRIPS = 2 + 3 * 3 + 1 + 2 + 1
_LOCK_ROUND_TRIPS = 3 + 2 + 2
DREAM_STORE_IO_SECONDS = (
    GUARD_BUDGET_SECONDS
    + _RECORD_ROUND_TRIPS * RECORD_WRITE_TIMEOUT_SECONDS
    + _LOCK_ROUND_TRIPS * LOCK_CHECK_TIMEOUT_SECONDS
)

# Per-user marker stamped after a successful (non-skipped) sync apply so
# the next nightly pass can skip all three LLM phases when no new episode
# landed since. Single key — prod Redis runs in cluster mode (see
# locks.py), so no multi-key primitives. The 35-day TTL comfortably
# outlives the 14-day episode window; an expired or missing marker just
# means one extra full pass (fail-open). The batch path does NOT stamp
# this marker yet — batch users simply never benefit from the skip.
# The key is ``MemoryScope.redis_key("last_completed")``.
LAST_COMPLETED_TTL_SECONDS = 35 * 24 * 60 * 60


_Output = TypeVar("_Output", bound=BaseModel)


class _PassInference(BaseModel):
    """What the phases of one sync pass share when they call the model: who
    the pass runs for, its id (every phase's correlation id) and the config
    whose fast models the phases run on."""

    model_config = ConfigDict(frozen=True)

    scope: InferenceScope
    pass_id: str
    config: ChatConfig


async def _run_consolidate(
    run: _PassInference, input_bundle: DreamInput
) -> tuple[ConsolidationOutput, PhaseUsage]:
    """First step: merge near-duplicate recent facts into canonical statements."""
    return await _run_phase(
        run,
        "consolidate",
        build_consolidate_prompt(input_bundle),
        ConsolidationOutput,
        temperature=CONSOLIDATE_TEMP,
        max_output_tokens=CONSOLIDATE_MAX_TOKENS,
        timeout_seconds=CONSOLIDATE_TIMEOUT_SECONDS,
    )


async def _run_recombine(
    run: _PassInference,
    input_bundle: DreamInput,
    consolidated: ConsolidationOutput,
) -> tuple[RecombinationOutput, PhaseUsage]:
    """Second step: propose novel connections + weak-link findings."""
    return await _run_phase(
        run,
        "recombine",
        build_recombine_prompt(input_bundle, consolidated.model_dump_json()),
        RecombinationOutput,
        temperature=RECOMBINE_TEMP,
        max_output_tokens=RECOMBINE_MAX_TOKENS,
        timeout_seconds=RECOMBINE_TIMEOUT_SECONDS,
    )


async def _run_sanitize(
    run: _PassInference,
    input_bundle: DreamInput,
    consolidated: ConsolidationOutput,
    recombined: RecombinationOutput,
) -> tuple[DreamOperations, PhaseUsage]:
    """Third step: gate writes/proposals/demotions before apply.py runs."""
    return await _run_phase(
        run,
        "sanitize",
        build_sanitize_prompt(
            input_bundle,
            consolidated.model_dump_json(),
            recombined.model_dump_json(),
        ),
        DreamOperations,
        temperature=SANITIZE_TEMP,
        max_output_tokens=SANITIZE_MAX_TOKENS,
        timeout_seconds=SANITIZE_TIMEOUT_SECONDS,
    )


async def _run_phase(
    run: _PassInference,
    phase: DreamPhase,
    messages: list[dict[str, str]],
    response_model: type[_Output],
    *,
    temperature: float,
    max_output_tokens: int,
    timeout_seconds: float,
) -> tuple[_Output, PhaseUsage]:
    """One phase's call on the pass's route, traced, with its cost recorded.

    The usage comes back priced: the provider's cost when it reported one
    (OpenRouter's ``usage.cost``, what we were billed), else the model's
    catalog list rate; unknown when the response reported no usage at all.

    Raises ``InferenceError`` when the phase got no usable answer. Failed
    phases are absorbed; their usage remains on the error for tracing.
    """
    job = phase_job(phase, run.pass_id, timeout_seconds=timeout_seconds)
    route = resolve_route(run.scope, job, config=run.config)
    ctx = InferenceContext(scope=run.scope, job=job, route=route)
    async with trace(ctx) as call:
        completion = await structured_complete(
            call.ctx,
            messages,
            response_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        usage = await record_phase_cost(call.ctx, completion.usage)
        call.usage = usage
    return completion.value, phase_usage(phase, usage)


async def _read_last_completed_marker(scope: MemoryScope) -> datetime | None:
    """When the user's last dream pass completed, or ``None``.

    Best-effort: a Redis error or an unparseable value fails open
    (``None`` ⇒ the pass runs) — the marker only exists to save LLM
    spend, so it must never block a dream.
    """
    # Lazy import matching locks.py — keeps the module cheap to import
    # in tests that mock redis.
    from backend.data.redis_client import get_redis_async

    user_id = scope.owner_user_id
    try:
        redis = await get_redis_async()
        raw = await redis.get(scope.redis_key("last_completed"))
    except Exception:
        logger.warning(
            "Failed to read dream last-completed marker for user %s — "
            "running the pass",
            user_id[:12],
            exc_info=True,
        )
        return None
    if raw is None:
        return None
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        marker = datetime.fromisoformat(str(raw))
    except (UnicodeDecodeError, ValueError):
        logger.warning(
            "Unparseable dream last-completed marker for user %s — running the pass",
            user_id[:12],
        )
        return None
    return marker if marker.tzinfo else marker.replace(tzinfo=timezone.utc)


async def _stamp_last_completed_marker(scope: MemoryScope, as_of: datetime) -> None:
    """Record the upper bound of the episode window the pass consolidated.

    ``as_of`` must be the gather snapshot time (``DreamInput.window_end``),
    NOT apply-completion time: the three LLM phases + apply take minutes,
    and an episode enqueued in that window was absent from the bundle —
    stamping "now" would mark it as already consolidated and skip it until
    the next genuinely-new episode (or the marker TTL).

    Best-effort: a failed stamp only costs one extra full pass on the
    next nightly tick.
    """
    from backend.data.redis_client import get_redis_async

    try:
        redis = await get_redis_async()
        await redis.set(
            scope.redis_key("last_completed"),
            as_of.isoformat(),
            ex=LAST_COMPLETED_TTL_SECONDS,
        )
    except Exception:
        logger.warning(
            "Failed to stamp dream last-completed marker for user %s",
            scope.owner_user_id[:12],
            exc_info=True,
        )


def _has_new_episodes_since(episodes: list[EpisodeRow], marker: datetime) -> bool:
    """True when any *user* episode is newer than the last-completed marker.

    Dream-authored episodes are excluded: a productive pass enqueues its
    own writes with ``valid_at = now()`` (after the stamped ``window_end``),
    so counting them would make every subsequent nightly run a paid no-op
    that only re-reads its own output.

    An episode with no parseable timestamp counts as new — we can't
    prove it's old, and skipping a pass we owed the user is worse than
    running one we didn't.
    """
    for episode in episodes:
        if is_dream_authored_episode(episode):
            continue
        episode_at = parse_episode_timestamp(episode)
        if episode_at is None or episode_at > marker:
            return True
    return False


async def execute_dream_pass(
    user_id: str,
    *,
    status_id: str | None = None,
    expert_id: str | None = None,
    trigger: DreamTrigger = "cron",
    force: bool = False,
) -> DreamPassResult:
    """Public async entry point used by the scheduler + admin trigger.

    ``status_id`` is the JobStatus row id when the caller is the polling
    admin trigger: it lets the trigger re-run a pass the last-completed
    marker would skip, and on the batch route the callbacks advance that
    row as each phase lands. Which route a pass takes depends on the batch
    flag and the deployment's Anthropic key, not on the caller, so a pass
    without a ``status_id`` can take the batch route too; its callbacks
    then skip the JobStatus writes.

    ``trigger`` is what started the pass, as its ``DreamPass`` row records
    it: ``cron`` for the nightly job, ``admin`` for the admin triggers,
    ``eval`` for an eval run. ``force`` is the admin trigger's, see ``guard.py``.
    """
    return await _execute_dream_pass_async(
        user_id, status_id=status_id, expert_id=expert_id, trigger=trigger, force=force
    )


@attributed_usage
async def _execute_dream_pass_async(
    user_id: str,
    *,
    expert_id: str | None,
    config: ChatConfig | None = None,
    status_id: str | None = None,
    trigger: DreamTrigger = "cron",
    force: bool = False,
) -> DreamPassResult:
    config = config or ChatConfig()
    run = DreamPassRun.begin(user_id, await _route_for(user_id, config), force=force)
    try:
        scope = MemoryScope.build(user_id, expert_id)
        if not await start_pass(run, scope, trigger=trigger):
            # A hard forget finds only passes whose rows it can read: one
            # whose row may not be there must not read the graph.
            result = run.failure(pass_error("pass", "record_unconfirmed"))
            await record_sync_outcome(result)
            return result
        async with dream_lock(scope, token=run.lease_token) as handle:
            run.hold(handle)
            return await _run_locked(
                run, scope, handle, config=config, status_id=status_id
            )
    except DreamLockHeld:
        result = run.skipped("lock_held")
    except Exception as exc:  # pragma: no cover — last-resort guard
        logger.exception("Dream pass crashed for user %s: %s", user_id[:12], exc)
        result = run.failure(pass_error("pass", "crashed"))
    await record_sync_outcome(result)
    return result


async def _route_for(user_id: str, config: ChatConfig) -> ExecutionPath:
    """The route a pass takes.

    The async Anthropic batch path is gated by the
    ``DREAM_PASS_BATCH_ENABLED`` LD flag AND a direct Anthropic key (the
    native Batch API can't be reached via OpenRouter/subscription, so the
    key is a hard requirement). When the flag is off, dreams run on the
    synchronous baseline regardless of key presence — the flag lets the
    batch path ship dark and roll out per-cohort. When on, phase 1 submits
    via call_provider(execution_mode="batch"); the BatchExecutor polls and
    dream's batch_callbacks chain phases 2 → 3 + apply when results land
    (half the list price, see routing.batch_discount).

    ``transport_name`` short-circuits to sync_baseline for transports that
    can't honour a batch path (local backends have no batch API;
    subscription mode shouldn't dual-bill the user's Anthropic key when the
    chat layer is on Claude Code OAuth).
    """
    batch_enabled = await is_feature_enabled(Flag.DREAM_PASS_BATCH_ENABLED, user_id)
    return resolve_dream_execution_path(
        has_anthropic_key=bool(config.direct_anthropic_api_key),
        batch_processing_enabled=batch_enabled,
        transport_name=config.transport.name,
    )


async def _run_locked(
    run: DreamPassRun,
    scope: MemoryScope,
    lock_handle: DreamLockHandle,
    *,
    config: ChatConfig,
    status_id: str | None,
) -> DreamPassResult:
    """The pass while it holds its scope's lock. Never raises: an unexpected
    error becomes a failure carrying the usage the pass has run up.

    The outcome write is attempted here, inside the lock, so the next pass
    to take the lock normally finds this one's row closed. It is best-effort
    like every record write: one that fails or times out is dropped and the
    lock is released anyway, so a free lock can still have an open row
    (APPLYING, say) behind it until a later pass's guard or the reaper
    closes it."""
    try:
        result = await _dream(
            run, scope, lock_handle, config=config, status_id=status_id
        )
    except DreamLockLostError:
        logger.exception("Dream pass lost its lock for user %s", run.user_id[:12])
        result = run.failure(pass_error("apply", "lock_lost"))
    except PhaseChargeError:
        logger.exception(
            "Dream pass could not be charged for user %s", run.user_id[:12]
        )
        result = run.failure(pass_error("billing", "charge_failed"))
    except Exception as exc:
        logger.exception("Dream pass crashed for user %s: %s", run.user_id[:12], exc)
        result = run.failure(pass_error("pass", "crashed"))
    await record_sync_outcome(result)
    return result


async def _dream(
    run: DreamPassRun,
    scope: MemoryScope,
    lock_handle: DreamLockHandle,
    *,
    config: ChatConfig,
    status_id: str | None,
) -> DreamPassResult:
    """Guard, gather, then hand the pass to the batch route or run its three
    phases and apply them. A pass that ends early (a skip, a stop, a billing
    or phase failure) comes back as the result it ended with."""
    try:
        await guard_dream_pass(run, scope)
        input_bundle = await _gather(run, scope, config=config, status_id=status_id)
        if run.execution_path == "anthropic_batch":
            # Phase 1 goes to Anthropic's Messages Batches API with the
            # output tool, and the pass returns at once. The BatchExecutor
            # polls; dream's batch_callbacks chain phases 2 → 3 + apply as
            # each result lands. Total latency is provider-driven (typically
            # <30min, hard cap 24h per phase batch via
            # BatchExecutor.MAX_BATCH_LIFETIME_SECONDS).
            return await submit_dream_pass_batch(
                run,
                config=config,
                input_bundle=input_bundle,
                status_id=status_id,
                lock_handle=lock_handle,
            )
        inference = _PassInference(
            scope=InferenceScope(user_id=run.user_id, expert_id=scope.expert_id),
            pass_id=run.pass_id,
            config=config,
        )
        sanitized = await _run_phases(run, inference, input_bundle)
        return await _apply(run, scope, lock_handle, sanitized, input_bundle)
    except PassEnded as ended:
        return ended.result


async def _gather(
    run: DreamPassRun,
    scope: MemoryScope,
    *,
    config: ChatConfig,
    status_id: str | None,
) -> DreamInput:
    """Check the budget, gather the input, and end a pass with nothing to do.

    The billing check runs inside the lock so a paywalled user doesn't burn
    the slot for an eligible concurrent pass on a shared FalkorDB. Raises
    ``PassEnded`` with the skip or failure that ends the pass early.
    """
    budget_ok, budget_skip = await check_dream_budget(run.user_id, config=config)
    if not budget_ok:
        if budget_skip == "rate_limit_unavailable":
            raise PassEnded(
                run.failure(pass_error("billing", "rate_limit_unavailable"))
            )
        raise PassEnded(run.skipped(budget_skip or "insufficient_credits"))
    input_bundle = await gather_dream_input(scope)
    if not input_bundle.episodes and not input_bundle.facts:
        # Nothing to consolidate — skipped so the admin UI can render
        # "nothing to dream about yet".
        raise PassEnded(run.skipped("no_input"))
    if await _nothing_new_since_last_pass(run, scope, input_bundle, status_id):
        raise PassEnded(run.skipped("no_new_activity"))
    if not await record_gathered(run.pass_id, input_bundle):
        # What the pass read must be on its row before any model call: a
        # hard forget marks a closed pass by it. A row that refused it was
        # closed by a stop, whose error the pass ends with.
        stopped = await stopped_error(run.pass_id)
        raise PassEnded(run.failure(stopped or pass_error("gather", "record_failed")))
    return input_bundle


async def _nothing_new_since_last_pass(
    run: DreamPassRun,
    scope: MemoryScope,
    input_bundle: DreamInput,
    status_id: str | None,
) -> bool:
    """No NEW activity since the last completed pass: every episode in the
    bundle predates the marker, so re-running all three LLM phases would
    only re-chew already-consolidated material (and, before the empty-pass
    guard in apply.py, manufacture an empty dream chat). Marker read is
    best-effort: missing, unparseable, or Redis-down all mean "run the pass".

    Manual admin triggers (the only callers that set status_id) bypass the
    marker: "dream now" is the memory-debugging tool for re-running a pass
    after prompt/flag/model changes, and a silent no_new_activity skip
    would neuter it for up to the marker's 35-day TTL.
    """
    last_completed = (
        await _read_last_completed_marker(scope) if status_id is None else None
    )
    if last_completed is None or _has_new_episodes_since(
        input_bundle.episodes, last_completed
    ):
        return False
    logger.info(
        "Dream pass %s skipped for user %s — no episodes newer "
        "than last completed pass at %s",
        run.pass_id,
        run.user_id[:12],
        last_completed.isoformat(),
    )
    return True


async def _run_phases(
    run: DreamPassRun, inference: _PassInference, input_bundle: DreamInput
) -> DreamOperations:
    """Consolidate, recombine and sanitize in turn, recording each phase's
    usage on the run and its output on the record as it lands."""
    consolidated = await _phase(
        run, "consolidate", lambda: _run_consolidate(inference, input_bundle)
    )
    recombined = await _phase(
        run, "recombine", lambda: _run_recombine(inference, input_bundle, consolidated)
    )
    return await _phase(
        run,
        "sanitize",
        lambda: _run_sanitize(inference, input_bundle, consolidated, recombined),
    )


async def _phase(
    run: DreamPassRun,
    phase: DreamPhase,
    call: Callable[[], Awaitable[tuple[_Output, PhaseUsage]]],
) -> _Output:
    """One phase's output. A phase with no usable answer ends the pass
    (``PassEnded``) with the completed phases' usage. Failed inference
    attempts are absorbed. A completed phase whose charge failed
    (``PhaseChargeError``) joins that usage too, then its error ends the
    pass as a crash does."""
    await checkpoint(run, phase)
    try:
        output, usage = await call()
    except InferenceError as exc:
        logger.warning("Dream pass %s: %s failed: %s", run.pass_id, phase, exc)
        raise PassEnded(run.failure(pass_error(phase, exc.code))) from exc
    except PhaseChargeError as exc:
        run.phases.append(phase_usage(phase, exc.usage))
        raise
    run.phases.append(usage)
    await record_phase_output(run.pass_id, phase, output)
    return output


async def _apply(
    run: DreamPassRun,
    scope: MemoryScope,
    lock_handle: DreamLockHandle,
    sanitized: DreamOperations,
    input_bundle: DreamInput,
) -> DreamPassResult:
    """Clamp the operations, make the last checks, apply and stamp the marker."""
    ops = clamp_pass_operations(sanitized, input_bundle)
    await record_applying(run.pass_id, ops)
    lease = await admit_sync_apply(run, scope, lock_handle)
    apply_stats = await apply_operations(
        scope,
        run.pass_id,
        ops,
        known_fact_uuids=input_bundle.known_fact_uuids,
        known_episode_uuids=input_bundle.known_episode_uuids,
        fact_scopes=fact_scopes(input_bundle),
        lock_handle=lock_handle,
        lease=lease,
    )
    # Apply succeeded (even as a no-op) — stamp the marker so the next nightly
    # pass can skip when nothing new has landed. Stamped with the gather-window
    # end so episodes that arrived mid-pass still count as new next time. Sync
    # path only: batch apply runs hours later in batch_callbacks, which doesn't
    # stamp yet.
    await _stamp_last_completed_marker(scope, input_bundle.window_end)
    return _applied_result(run, apply_stats, ops)


def _applied_result(
    run: DreamPassRun,
    apply_stats: dict[str, int | str | IngestionDrainStatus | DreamOperationsSnapshot],
    ops: DreamOperations,
) -> DreamPassResult:
    snapshot = apply_stats.get("snapshot")
    raw_session_id = apply_stats.get("session_id")

    def _as_int(key: str) -> int:
        v = apply_stats.get(key, 0)
        return int(v) if isinstance(v, (int, str)) and v else 0

    return DreamPassResult(
        user_id=run.user_id,
        pass_id=run.pass_id,
        started_at=run.started_at,
        completed_at=datetime.now(timezone.utc),
        elapsed_seconds=run.elapsed_seconds(),
        execution_path=run.execution_path,
        dropped_forgotten=_as_int("dropped_forgotten"),
        pending_writes=_as_int("pending_writes"),
        failed_writes=_as_int("failed_writes"),
        consolidated_count=_as_int("consolidated_count"),
        proposal_count=_as_int("proposal_count"),
        demotion_count=_as_int("demotion_count"),
        entity_invalidation_count=_as_int("entity_invalidation_count"),
        provenance_pending=_as_int("provenance_pending"),
        uncited_writes_dropped=_as_int("uncited_writes_dropped"),
        cross_scope_citations_dropped=_as_int("cross_scope_citations_dropped"),
        protected_demotions=_as_int("protected_demotions"),
        indeterminate_demotion_writes=_as_int("indeterminate_demotion_writes"),
        # Only apply's own False marks the count unconfirmed.
        demotion_accounting_complete=(
            apply_stats.get("demotion_accounting_complete") is not False
        ),
        summary_for_user=ops.summary_for_user,
        # Fail-closed: a missing/malformed drain flag reads as
        # ``timed_out`` (writes at risk), never a confirmed drain.
        ingestion_drain_status=drain_status_from_stats(apply_stats),
        # ``None`` (key absent) on an empty pass — apply skipped the
        # dream session entirely, so there is no id to surface.
        dream_session_id=(raw_session_id if isinstance(raw_session_id, str) else None),
        operations=(
            snapshot if isinstance(snapshot, DreamOperationsSnapshot) else None
        ),
        usage=run.usage(),
    )
