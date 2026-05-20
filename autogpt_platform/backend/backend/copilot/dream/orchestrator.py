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
from .fetch import DreamInput, gather_dream_input
from .llm import DreamLLMError, structured_completion
from .locks import (
    DEFAULT_LOCK_TTL_SECONDS,
    LOCAL_LOCK_TTL_SECONDS,
    DreamLockHeld,
    dream_lock,
)
from .prompts import (
    MAX_DEMOTIONS_PER_PASS,
    MAX_PROPOSALS_PER_PASS,
    MAX_WRITES_PER_PASS,
    build_phase_1_prompt,
    build_phase_2_prompt,
    build_phase_3_prompt,
)
from .routing import ExecutionPath, resolve_dream_execution_path
from .schemas import (
    ConsolidationOutput,
    DreamOperations,
    DreamPassResult,
    RecombinationOutput,
)

logger = logging.getLogger(__name__)


# Phase temperatures + max output tokens, matching ``dream/p0-spec.md`` §2.
PHASE_1_TEMP = 0.2
PHASE_2_TEMP = 0.9
PHASE_3_TEMP = 0.0
PHASE_1_MAX_TOKENS = 4096
PHASE_2_MAX_TOKENS = 8192
PHASE_3_MAX_TOKENS = 8192


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


async def _run_phase_1(
    config: ChatConfig, input_bundle: DreamInput
) -> ConsolidationOutput:
    messages = build_phase_1_prompt(input_bundle)
    return await structured_completion(
        model=config.fast_standard_model,
        messages=messages,
        response_model=ConsolidationOutput,
        temperature=PHASE_1_TEMP,
        max_output_tokens=PHASE_1_MAX_TOKENS,
    )


async def _run_phase_2(
    config: ChatConfig,
    input_bundle: DreamInput,
    phase_1: ConsolidationOutput,
) -> RecombinationOutput:
    messages = build_phase_2_prompt(input_bundle, phase_1.model_dump_json())
    return await structured_completion(
        model=config.fast_advanced_model,
        messages=messages,
        response_model=RecombinationOutput,
        temperature=PHASE_2_TEMP,
        max_output_tokens=PHASE_2_MAX_TOKENS,
    )


async def _run_phase_3(
    config: ChatConfig,
    input_bundle: DreamInput,
    phase_1: ConsolidationOutput,
    phase_2: RecombinationOutput,
) -> DreamOperations:
    messages = build_phase_3_prompt(
        input_bundle,
        phase_1.model_dump_json(),
        phase_2.model_dump_json(),
    )
    return await structured_completion(
        model=config.fast_standard_model,
        messages=messages,
        response_model=DreamOperations,
        temperature=PHASE_3_TEMP,
        max_output_tokens=PHASE_3_MAX_TOKENS,
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

            try:
                phase_1 = await _run_phase_1(config, input_bundle)
            except DreamLLMError as exc:
                return _failure_result(
                    user_id,
                    pass_id,
                    started_at,
                    monotonic_start,
                    execution_path,
                    f"phase_1: {exc}",
                )

            try:
                phase_2 = await _run_phase_2(config, input_bundle, phase_1)
            except DreamLLMError as exc:
                return _failure_result(
                    user_id,
                    pass_id,
                    started_at,
                    monotonic_start,
                    execution_path,
                    f"phase_2: {exc}",
                )

            try:
                phase_3 = await _run_phase_3(config, input_bundle, phase_1, phase_2)
            except DreamLLMError as exc:
                return _failure_result(
                    user_id,
                    pass_id,
                    started_at,
                    monotonic_start,
                    execution_path,
                    f"phase_3: {exc}",
                )

            ops = _clamp_operations(phase_3)
            apply_stats = await apply_operations(user_id, pass_id, ops)

            completed_at = datetime.now(timezone.utc)
            return DreamPassResult(
                user_id=user_id,
                pass_id=pass_id,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_seconds=(asyncio.get_event_loop().time() - monotonic_start),
                execution_path=execution_path,
                consolidated_count=int(apply_stats.get("consolidated_count", 0) or 0),
                proposal_count=int(apply_stats.get("proposal_count", 0) or 0),
                demotion_count=int(apply_stats.get("demotion_count", 0) or 0),
                entity_invalidation_count=int(
                    apply_stats.get("entity_invalidation_count", 0) or 0
                ),
                summary_for_user=ops.summary_for_user,
                dream_session_id=str(apply_stats.get("session_id") or ""),
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
) -> DreamPassResult:
    return DreamPassResult(
        user_id=user_id,
        pass_id=pass_id,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        elapsed_seconds=asyncio.get_event_loop().time() - monotonic_start,
        execution_path=execution_path,
        error=error,
    )


async def execute_dream_pass(user_id: str) -> DreamPassResult:
    """Public async entry point used by the scheduler + admin trigger."""
    return await _execute_dream_pass_async(user_id)
