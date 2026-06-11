"""Three-phase dream-pass orchestrator (sync_baseline path).

Walks a user's recent memory window through the consolidate → recombine
→ sanitize pipeline, then applies the sanitizer's ``DreamOperations``
to Graphiti + Postgres.

Slice 1 of P-0 deliberately bypasses Anthropic batch — every phase
calls the OpenRouter-fronted OpenAI-compat client. The batch path
slots in below this layer (`routing.py` returns ``"batch"``) in a
future PR.

The orchestrator never raises out — every failure becomes a
``DreamPassResult`` with ``error`` set, so the admin trigger always
gets a structured response back.
"""

from __future__ import annotations

import asyncio
import logging
import uuid as uuidlib
from datetime import datetime, timezone

from backend.copilot.config import ChatConfig
from backend.util.feature_flag import Flag, is_feature_enabled

from .apply import apply_operations
from .billing import check_dream_budget, record_phase_cost
from .fetch import DreamInput, gather_dream_input
from .llm import (
    CompletionUsage,
    DreamLLMError,
    StructuredCompletion,
    structured_completion,
)
from .locks import (
    BATCH_LOCK_TTL_SECONDS,
    DEFAULT_LOCK_TTL_SECONDS,
    LOCAL_LOCK_TTL_SECONDS,
    DreamLockHandle,
    DreamLockHeld,
    dream_lock,
)
from .model_pricing import compute_cost_usd, execution_path_discount
from .prompts import (
    MAX_DEMOTIONS_PER_PASS,
    MAX_PROPOSALS_PER_PASS,
    MAX_WRITES_PER_PASS,
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
    DreamPassUsage,
    PhaseUsage,
    RecombinationOutput,
)

logger = logging.getLogger(__name__)


# Per-step temperature + max output token budgets. Spec ref:
# ``dream/p0-spec.md`` §2. Named by what the step does rather than its
# position; the consolidation→recombination→sanitization order is
# enforced by the call graph below.
CONSOLIDATE_TEMP = 0.2
RECOMBINE_TEMP = 0.9
SANITIZE_TEMP = 0.0
CONSOLIDATE_MAX_TOKENS = 4096
# Recombine + sanitize emit list-heavy JSON (up to 30 writes + 20
# proposals + 10 demotions, each with uuid arrays). At 8192 the
# response truncates mid-array and the JSON-balanced-brace extractor
# returns None, failing the whole phase. 16384 covers worst-case +
# headroom; cost impact is bounded by the per-phase caps anyway.
RECOMBINE_MAX_TOKENS = 16384
SANITIZE_MAX_TOKENS = 16384

# Entity invalidation is the most destructive op the sanitizer can emit:
# each one single-hop demotes EVERY :RELATES_TO edge on the entity, so a
# hub entity multiplies the blast radius far past the demotion caps. We
# cap the *count* of invalidations per pass here; the per-entity edge
# blast radius is bounded by the ``DREAM_PASS_INVALIDATE_ENTITY`` LD flag
# (staged rollout, off by default) plus the single-hop-only guarantee in
# ``invalidate_entity_direct_neighbors`` (no multi-hop propagation).
# Deliberately NOT degree-aware — fetching entity degrees is async graph
# work that doesn't belong in this sync clamp.
MAX_ENTITY_INVALIDATIONS_PER_PASS = 2


def _resolve_lock_ttl(transport_is_local: bool) -> int:
    return LOCAL_LOCK_TTL_SECONDS if transport_is_local else DEFAULT_LOCK_TTL_SECONDS


def _clamp_operations(
    ops: DreamOperations,
    active_fact_count: int,
    known_fact_uuids: set[str] | None = None,
) -> DreamOperations:
    """Hard-trim oversized phase 3 outputs.

    Phase 3's prompt asks for these caps but the model can still
    over-emit. The orchestrator enforces them in code so apply.py
    never writes more than the policy allows.

    Demotions carry a second ceiling — 5% of the active fact set — so a
    single pass can never wipe a meaningful fraction of a user's memory
    even if the absolute ``MAX_DEMOTIONS_PER_PASS`` cap would allow it.
    The 5% ceiling floors at 1 when there is at least one active fact:
    small graphs (< 20 facts) would otherwise round to a cap of 0 and
    never get even a single contradicted fact demoted.
    ``active_fact_count < 0`` means the count is unknown (the batch path
    lost its persisted input bundle); fall back to the absolute cap only
    rather than silently dropping every demotion.

    When ``known_fact_uuids`` is provided, demotions targeting uuids
    outside it are dropped BEFORE the cap slice — otherwise a
    hallucinated uuid at the head of the model's list consumes a cap
    slot (the entire budget on a floor-of-1 small graph) and displaces
    a valid demotion that apply.py would have accepted. ``None`` skips
    the pre-filter; apply.py's idempotent known-uuid filter remains the
    security chokepoint either way.

    Entity invalidations are count-capped at
    ``MAX_ENTITY_INVALIDATIONS_PER_PASS``; see the constant's comment
    for why the per-entity edge blast radius is bounded elsewhere (LD
    flag + single-hop guarantee), not here.
    """
    demotions = ops.demotions
    if known_fact_uuids is not None:
        demotions = [d for d in demotions if d.edge_uuid in known_fact_uuids]
        dropped = len(ops.demotions) - len(demotions)
        if dropped:
            logger.warning(
                "Dream clamp: dropped %d demotion(s) targeting edge uuids "
                "outside known_fact_uuids before applying the demotion cap",
                dropped,
            )
    demotion_cap = MAX_DEMOTIONS_PER_PASS
    if active_fact_count == 0:
        demotion_cap = 0
    elif active_fact_count > 0:
        demotion_cap = min(MAX_DEMOTIONS_PER_PASS, max(1, active_fact_count * 5 // 100))
    return DreamOperations(
        writes=ops.writes[:MAX_WRITES_PER_PASS],
        proposals=ops.proposals[:MAX_PROPOSALS_PER_PASS],
        demotions=demotions[:demotion_cap],
        entity_invalidations=ops.entity_invalidations[
            :MAX_ENTITY_INVALIDATIONS_PER_PASS
        ],
        summary_for_user=ops.summary_for_user,
    )


async def _run_consolidate(
    config: ChatConfig, input_bundle: DreamInput
) -> StructuredCompletion[ConsolidationOutput]:
    """First step: merge near-duplicate recent facts into canonical statements."""
    messages = build_consolidate_prompt(input_bundle)
    return await structured_completion(
        model=config.fast_standard_model,
        messages=messages,
        response_model=ConsolidationOutput,
        temperature=CONSOLIDATE_TEMP,
        max_output_tokens=CONSOLIDATE_MAX_TOKENS,
    )


async def _run_recombine(
    config: ChatConfig,
    input_bundle: DreamInput,
    consolidated: ConsolidationOutput,
) -> StructuredCompletion[RecombinationOutput]:
    """Second step: propose novel connections + weak-link findings."""
    messages = build_recombine_prompt(input_bundle, consolidated.model_dump_json())
    return await structured_completion(
        model=config.fast_advanced_model,
        messages=messages,
        response_model=RecombinationOutput,
        temperature=RECOMBINE_TEMP,
        max_output_tokens=RECOMBINE_MAX_TOKENS,
    )


async def _run_sanitize(
    config: ChatConfig,
    input_bundle: DreamInput,
    consolidated: ConsolidationOutput,
    recombined: RecombinationOutput,
) -> StructuredCompletion[DreamOperations]:
    """Third step: gate writes/proposals/demotions before apply.py runs."""
    messages = build_sanitize_prompt(
        input_bundle,
        consolidated.model_dump_json(),
        recombined.model_dump_json(),
    )
    return await structured_completion(
        model=config.fast_standard_model,
        messages=messages,
        response_model=DreamOperations,
        temperature=SANITIZE_TEMP,
        max_output_tokens=SANITIZE_MAX_TOKENS,
    )


def _phase_usage_from_completion(
    phase: str, completion_usage: CompletionUsage, execution_path: ExecutionPath
) -> PhaseUsage:
    """Build a ``PhaseUsage`` from a raw ``CompletionUsage``, computing
    cost from the rate card when the provider didn't supply one.

    Provider-supplied cost (OpenRouter ``usage.cost``) wins when
    present; otherwise we fall back to ``model_pricing.compute_cost_usd``.
    Either way the execution-path discount has been applied: OpenRouter
    spot prices are already post-discount (they ARE the rate we paid),
    and the rate-card fallback explicitly multiplies the discount in.
    """
    cost = completion_usage.cost_usd
    if cost is None:
        cost = compute_cost_usd(
            model=completion_usage.model,
            input_tokens=completion_usage.input_tokens,
            output_tokens=completion_usage.output_tokens,
            cache_read_tokens=completion_usage.cache_read_tokens,
            cache_creation_tokens=completion_usage.cache_creation_tokens,
            execution_path=execution_path,
        )
    return PhaseUsage(
        phase=phase,  # type: ignore[arg-type]
        model=completion_usage.model,
        input_tokens=completion_usage.input_tokens,
        output_tokens=completion_usage.output_tokens,
        cache_read_tokens=completion_usage.cache_read_tokens,
        cache_creation_tokens=completion_usage.cache_creation_tokens,
        cost_usd=cost,
    )


def _aggregate_usage(
    phases: list[PhaseUsage], execution_path: ExecutionPath
) -> DreamPassUsage:
    """Roll up per-phase usage into a ``DreamPassUsage``.

    ``total_cost_usd`` is None when any single phase had unknown cost
    so we never silently bill at a partial figure.
    """
    total_cost: float | None = 0.0
    for p in phases:
        if p.cost_usd is None:
            total_cost = None
            break
        total_cost += p.cost_usd
    return DreamPassUsage(
        phases=phases,
        total_input_tokens=sum(p.input_tokens for p in phases),
        total_output_tokens=sum(p.output_tokens for p in phases),
        total_cache_read_tokens=sum(p.cache_read_tokens for p in phases),
        total_cache_creation_tokens=sum(p.cache_creation_tokens for p in phases),
        total_cost_usd=total_cost,
        discount_applied=execution_path_discount(execution_path),
    )


async def _execute_dream_pass_async(
    user_id: str,
    *,
    transport_is_local: bool = False,
    config: ChatConfig | None = None,
    status_id: str | None = None,
) -> DreamPassResult:
    config = config or ChatConfig()
    pass_id = str(uuidlib.uuid4())
    started_at = datetime.now(timezone.utc)
    monotonic_start = asyncio.get_event_loop().time()

    has_anthropic_key = bool(config.direct_anthropic_api_key)
    # Step 5: the async Anthropic batch path is gated by the
    # ``DREAM_PASS_BATCH_ENABLED`` LD flag AND a direct Anthropic key (the
    # native Batch API can't be reached via OpenRouter/subscription, so the
    # key is a hard requirement). When the flag is off, dreams run on the
    # synchronous baseline regardless of key presence — the flag lets the
    # batch path ship dark and roll out per-cohort. When on, phase 1 submits
    # via call_provider(execution_mode="batch"); the BatchExecutor polls and
    # dream's batch_callbacks chain phases 2 → 3 + apply when results land
    # (~50% off the rate card, see model_pricing.execution_path_discount).
    #
    # ``transport_name`` short-circuits to sync_baseline for transports that
    # can't honour a batch path (local backends have no batch API;
    # subscription mode shouldn't dual-bill the user's Anthropic key when the
    # chat layer is on Claude Code OAuth).
    batch_enabled = await is_feature_enabled(Flag.DREAM_PASS_BATCH_ENABLED, user_id)
    execution_path: ExecutionPath = resolve_dream_execution_path(
        has_anthropic_key=has_anthropic_key,
        batch_processing_enabled=batch_enabled,
        transport_name=config.transport.name,
    )

    ttl = _resolve_lock_ttl(transport_is_local)
    try:
        async with dream_lock(user_id, ttl_seconds=ttl) as dream_lock_handle:
            # Pre-flight billing check. Runs inside the lock so a
            # paywalled user doesn't burn the slot for an eligible
            # concurrent pass on a shared FalkorDB.
            budget_ok, budget_skip = await check_dream_budget(user_id, config=config)
            if not budget_ok:
                if budget_skip == "rate_limit_unavailable":
                    return _failure_result(
                        user_id,
                        pass_id,
                        started_at,
                        monotonic_start,
                        execution_path,
                        f"billing: {budget_skip}",
                    )
                return DreamPassResult(
                    user_id=user_id,
                    pass_id=pass_id,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    elapsed_seconds=(asyncio.get_event_loop().time() - monotonic_start),
                    execution_path=execution_path,
                    skipped=True,
                    skip_reason=budget_skip or "insufficient_credits",
                )

            input_bundle = await gather_dream_input(user_id)

            if not input_bundle.episodes and not input_bundle.facts:
                # Nothing to consolidate — early-return as skipped so the
                # admin UI can render "nothing to dream about yet".
                return DreamPassResult(
                    user_id=user_id,
                    pass_id=pass_id,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    elapsed_seconds=(asyncio.get_event_loop().time() - monotonic_start),
                    execution_path=execution_path,
                    skipped=True,
                    skip_reason="no_input",
                )

            # ---- Anthropic batch path -----------------------------------
            #
            # When routing picks ``anthropic_batch`` we persist the
            # input bundle, submit phase 1 to Anthropic's Messages
            # Batches API with forced tool_use structured output, and
            # return ``status="submitted"`` immediately. The BatchExecutor
            # polls; dream's batch_callbacks chain phases 2 → 3 + apply
            # when each phase result lands. Total latency is provider-
            # driven (typically <30min, hard cap 24h via
            # BatchExecutor.MAX_BATCH_LIFETIME_SECONDS); the user sees
            # JobStatus.current_phase advance through consolidate →
            # recombine → sanitize → complete.
            if execution_path == "anthropic_batch":
                return await _submit_dream_pass_batch(
                    user_id=user_id,
                    pass_id=pass_id,
                    started_at=started_at,
                    monotonic_start=monotonic_start,
                    execution_path=execution_path,
                    config=config,
                    input_bundle=input_bundle,
                    status_id=status_id,
                    dream_lock_handle=dream_lock_handle,
                )

            step_usages: list[PhaseUsage] = []

            try:
                consolidate_completion = await _run_consolidate(config, input_bundle)
            except DreamLLMError as exc:
                return _failure_result(
                    user_id,
                    pass_id,
                    started_at,
                    monotonic_start,
                    execution_path,
                    f"consolidate: {exc}",
                    usage=_aggregate_usage(step_usages, execution_path),
                )
            step_usages.append(
                _phase_usage_from_completion(
                    "consolidate", consolidate_completion.usage, execution_path
                )
            )
            await record_phase_cost(
                user_id=user_id,
                pass_id=pass_id,
                phase_usage=step_usages[-1],
                execution_path=execution_path,
            )
            consolidated = consolidate_completion.value

            try:
                recombine_completion = await _run_recombine(
                    config, input_bundle, consolidated
                )
            except DreamLLMError as exc:
                return _failure_result(
                    user_id,
                    pass_id,
                    started_at,
                    monotonic_start,
                    execution_path,
                    f"recombine: {exc}",
                    usage=_aggregate_usage(step_usages, execution_path),
                )
            step_usages.append(
                _phase_usage_from_completion(
                    "recombine", recombine_completion.usage, execution_path
                )
            )
            await record_phase_cost(
                user_id=user_id,
                pass_id=pass_id,
                phase_usage=step_usages[-1],
                execution_path=execution_path,
            )
            recombined = recombine_completion.value

            try:
                sanitize_completion = await _run_sanitize(
                    config, input_bundle, consolidated, recombined
                )
            except DreamLLMError as exc:
                return _failure_result(
                    user_id,
                    pass_id,
                    started_at,
                    monotonic_start,
                    execution_path,
                    f"sanitize: {exc}",
                    usage=_aggregate_usage(step_usages, execution_path),
                )
            step_usages.append(
                _phase_usage_from_completion(
                    "sanitize", sanitize_completion.usage, execution_path
                )
            )
            await record_phase_cost(
                user_id=user_id,
                pass_id=pass_id,
                phase_usage=step_usages[-1],
                execution_path=execution_path,
            )
            sanitized = sanitize_completion.value

            ops = _clamp_operations(
                sanitized,
                len(input_bundle.facts),
                known_fact_uuids=input_bundle.known_fact_uuids,
            )
            apply_stats = await apply_operations(
                user_id,
                pass_id,
                ops,
                known_fact_uuids=input_bundle.known_fact_uuids,
            )

            completed_at = datetime.now(timezone.utc)
            snapshot = apply_stats.get("snapshot")

            def _as_int(key: str) -> int:
                v = apply_stats.get(key, 0)
                return int(v) if isinstance(v, (int, str)) and v else 0

            return DreamPassResult(
                user_id=user_id,
                pass_id=pass_id,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_seconds=(asyncio.get_event_loop().time() - monotonic_start),
                execution_path=execution_path,
                consolidated_count=_as_int("consolidated_count"),
                proposal_count=_as_int("proposal_count"),
                demotion_count=_as_int("demotion_count"),
                entity_invalidation_count=_as_int("entity_invalidation_count"),
                summary_for_user=ops.summary_for_user,
                dream_session_id=str(apply_stats.get("session_id") or ""),
                operations=(
                    snapshot if isinstance(snapshot, DreamOperationsSnapshot) else None
                ),
                usage=_aggregate_usage(step_usages, execution_path),
            )

    except DreamLockHeld:
        return DreamPassResult(
            user_id=user_id,
            pass_id=pass_id,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            elapsed_seconds=(asyncio.get_event_loop().time() - monotonic_start),
            execution_path=execution_path,
            skipped=True,
            skip_reason="lock_held",
        )
    except Exception as exc:  # pragma: no cover — last-resort guard
        logger.exception("Dream pass crashed for user %s: %s", user_id[:12], exc)
        return _failure_result(
            user_id,
            pass_id,
            started_at,
            monotonic_start,
            execution_path,
            str(exc),
        )


def _failure_result(
    user_id: str,
    pass_id: str,
    started_at: datetime,
    monotonic_start: float,
    execution_path: ExecutionPath,
    error: str,
    usage: DreamPassUsage | None = None,
) -> DreamPassResult:
    """Build a failure ``DreamPassResult`` that still carries usage for
    the phases that completed before the error — billing has to charge
    for tokens we already paid for, even on partial failure."""
    return DreamPassResult(
        user_id=user_id,
        pass_id=pass_id,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        elapsed_seconds=asyncio.get_event_loop().time() - monotonic_start,
        execution_path=execution_path,
        error=error,
        usage=usage,
    )


async def execute_dream_pass(
    user_id: str, *, status_id: str | None = None
) -> DreamPassResult:
    """Public async entry point used by the scheduler + admin trigger.

    ``status_id`` is the optional JobStatus row id when the caller is
    the polling admin trigger (Step 2). The orchestrator threads it
    into the batch path so the BatchExecutor callbacks can update the
    user-visible status row as each phase lands. AgentProbe + other
    sync callers pass ``None`` and never hit the batch path.
    """
    return await _execute_dream_pass_async(user_id, status_id=status_id)


async def _submit_dream_pass_batch(
    *,
    user_id: str,
    pass_id: str,
    started_at: datetime,
    monotonic_start: float,
    execution_path: ExecutionPath,
    config: ChatConfig,
    input_bundle: DreamInput,
    status_id: str | None,
    dream_lock_handle: DreamLockHandle,
) -> DreamPassResult:
    """Submit phase 1 of the dream pass via Anthropic batch + return.

    The orchestrator hands control to the BatchExecutor here: phase 1
    (consolidate) is enqueued; phases 2 (recombine) and 3 (sanitize)
    fire from dream's batch_callbacks as each prior phase's result
    lands. Apply + cost log + JobStatus complete run when sanitize
    lands. This function only does the kickoff.

    The job_id link to JobStatus comes from the scheduler's
    ``execute_dream_pass_with_status`` wrapper (Step 2). When the
    orchestrator is invoked outside that path (e.g. AgentProbe eval),
    ``job_id`` is empty and the callback skips the JobStatus updates —
    apply still runs, cost still logs.
    """
    from .batch_submit import (
        persist_input_bundle,
        phase_models_for_config,
        submit_phase,
    )

    api_key = config.direct_anthropic_api_key
    if not api_key:
        # Shouldn't happen — routing.py only picks anthropic_batch when
        # the key is present. Guard for type-narrowing + future safety.
        return _failure_result(
            user_id,
            pass_id,
            started_at,
            monotonic_start,
            execution_path,
            "anthropic_batch: no Anthropic API key (routing bug)",
        )

    # Persist DreamInput so the per-phase callbacks can rebuild the
    # next phase's prompt without re-fetching from Postgres + FalkorDB.
    try:
        await persist_input_bundle(
            pass_id, input_bundle, lock_token=dream_lock_handle.token
        )
    except Exception as exc:
        return _failure_result(
            user_id,
            pass_id,
            started_at,
            monotonic_start,
            execution_path,
            f"anthropic_batch: input persist failed: {exc}",
        )

    # ``status_id`` ties the per-phase callback updates back to the
    # JobStatus row the admin trigger created. Empty string when no
    # caller wired it (AgentProbe eval, ad-hoc invocation) — the
    # callback skips status writes in that case but apply still runs.
    try:
        submission = await submit_phase(
            user_id=user_id,
            pass_id=pass_id,
            job_id=status_id or "",
            phase="consolidate",
            phase_models=phase_models_for_config(config),
            api_key=api_key,
            input_bundle=input_bundle,
        )
    except Exception as exc:
        return _failure_result(
            user_id,
            pass_id,
            started_at,
            monotonic_start,
            execution_path,
            f"anthropic_batch: phase 1 submit failed: {exc}",
        )

    completed_at = datetime.now(timezone.utc)
    logger.info(
        "Dream pass %s submitted via Anthropic batch=%s (phase=consolidate)",
        pass_id,
        submission.provider_batch_id,
    )
    # Phase 1 is enqueued — hand the dream lock to the batch callback so it
    # spans the full async lifetime (apply runs hours later). Extend the TTL
    # to the batch window first; the callback releases it on terminal/failure.
    await dream_lock_handle.extend(BATCH_LOCK_TTL_SECONDS)
    dream_lock_handle.disown()
    return DreamPassResult(
        user_id=user_id,
        pass_id=pass_id,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=asyncio.get_event_loop().time() - monotonic_start,
        execution_path=execution_path,
        skipped=False,
        # The pass is async-pending — no usage / ops / session_id yet.
        # The BatchExecutor + dream callbacks deliver those when phase
        # 3 lands and apply runs.
    )
