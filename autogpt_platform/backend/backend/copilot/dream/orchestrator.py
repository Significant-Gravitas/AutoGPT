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
    DEFAULT_LOCK_TTL_SECONDS,
    LOCAL_LOCK_TTL_SECONDS,
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
RECOMBINE_MAX_TOKENS = 8192
SANITIZE_MAX_TOKENS = 8192


def _resolve_lock_ttl(transport_is_local: bool) -> int:
    return LOCAL_LOCK_TTL_SECONDS if transport_is_local else DEFAULT_LOCK_TTL_SECONDS


def _clamp_operations(ops: DreamOperations) -> DreamOperations:
    """Hard-trim oversized phase 3 outputs.

    Phase 3's prompt asks for these caps but the model can still
    over-emit. The orchestrator enforces them in code so apply.py
    never writes more than the policy allows.
    """
    return DreamOperations(
        writes=ops.writes[:MAX_WRITES_PER_PASS],
        proposals=ops.proposals[:MAX_PROPOSALS_PER_PASS],
        demotions=ops.demotions[:MAX_DEMOTIONS_PER_PASS],
        entity_invalidations=ops.entity_invalidations,
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
) -> DreamPassResult:
    config = config or ChatConfig()
    pass_id = str(uuidlib.uuid4())
    started_at = datetime.now(timezone.utc)
    monotonic_start = asyncio.get_event_loop().time()

    has_anthropic_key = bool(config.direct_anthropic_api_key)
    execution_path: ExecutionPath = resolve_dream_execution_path(
        has_anthropic_key=has_anthropic_key,
        batch_processing_enabled=False,  # P-0.1 follow-up flips this on
    )

    ttl = _resolve_lock_ttl(transport_is_local)
    try:
        async with dream_lock(user_id, ttl_seconds=ttl):
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

            ops = _clamp_operations(sanitized)
            apply_stats = await apply_operations(user_id, pass_id, ops)

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


async def execute_dream_pass(user_id: str) -> DreamPassResult:
    """Public async entry point used by the scheduler + admin trigger."""
    return await _execute_dream_pass_async(user_id)
