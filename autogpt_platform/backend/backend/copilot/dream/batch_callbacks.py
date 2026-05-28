"""Dream-pass batch result handlers.

Registered with the BatchExecutor service under namespace
``"dream_pass"`` at module-import time. When the BatchExecutor finishes
polling a batch the dream pass submitted (via
``providers.call_provider(execution_mode="batch")``), it calls
``handle_dream_batch_result`` with the per-``custom_id`` result rows
plus the pending entry's ``payload`` (which the orchestrator wrote at
submit time: user_id, pass_id, job_id, dream_pass_id, phase mapping).

The handler:

  1. Parses each row's content into a Pydantic phase-output model
     (``ConsolidationOutput`` / ``RecombinationOutput`` /
     ``DreamOperations``). Empty / errored rows mark the phase failed.
  2. Once all three phases of a pass have landed, calls
     ``apply_operations`` to write the sanitized result to Graphiti +
     Postgres. (Today the dream pass orchestrator submits one batch
     per phase; the handler waits for all three batch entries to
     resolve before applying. Future grouping — all three phases in
     one batch — would land here too.)
  3. Logs per-phase token cost via ``record_phase_cost`` with
     ``execution_path="anthropic_batch"`` so the 50% discount is
     applied and the cost ledger reflects what actually shipped.
  4. Updates the JobStatus row (state: complete / errored,
     current_phase, result).

This module is dream-specific; future namespaces (P2 dedup, P3 self-
model refresh) each get their own callbacks module + namespace
registration.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.executor.batch_executor import PendingEntry
    from backend.util.llm.providers import BatchResultRow

logger = logging.getLogger(__name__)


NAMESPACE = "dream_pass"


# ---------------------------------------------------------------------------
# Handler entry point
# ---------------------------------------------------------------------------


async def handle_dream_batch_result(
    entry: "PendingEntry", rows: list["BatchResultRow"]
) -> None:
    """BatchExecutor entry — called once per finished batch.

    Today the dream orchestrator submits one batch per phase so each
    ``rows`` list has exactly one item. We aggregate across batches
    via Redis: phase results are written to
    ``dream:batch:phase_results:{pass_id}`` and the apply step fires
    when all three phases are present (or any phase errored — fail
    fast).
    """
    payload = entry.payload or {}
    user_id = payload.get("user_id") or ""
    pass_id = payload.get("pass_id") or ""
    job_id = payload.get("job_id") or ""
    phase_for_custom_id = payload.get("phase_for_custom_id") or {}

    if not user_id or not pass_id:
        logger.warning(
            "Dream batch handler missing user_id/pass_id — payload=%s; "
            "dropping batch %s",
            payload,
            entry.provider_batch_id,
        )
        return

    for row in rows:
        phase = phase_for_custom_id.get(row.custom_id)
        if not phase:
            logger.warning(
                "Unknown custom_id=%s for dream pass=%s — skipping",
                row.custom_id,
                pass_id,
            )
            continue
        await _record_phase_result(
            pass_id=pass_id,
            phase=phase,
            row=row,
        )

    state = await _read_phase_results(pass_id)
    if _has_terminal_error(state):
        await _finalize_errored(
            user_id=user_id,
            pass_id=pass_id,
            job_id=job_id,
            state=state,
        )
        return

    if not _all_phases_present(state):
        # Still waiting on later batches in this pass — leave the
        # accumulator in place. The next phase's handler call will
        # fire ``apply`` once the last one lands.
        return

    await _finalize_complete(
        user_id=user_id,
        pass_id=pass_id,
        job_id=job_id,
        state=state,
    )


# ---------------------------------------------------------------------------
# Per-pass Redis accumulator
# ---------------------------------------------------------------------------


def _accumulator_key(pass_id: str) -> str:
    return f"dream:batch:phase_results:{pass_id}"


# 6h TTL on the accumulator — matches the JobStatus TTL so a stalled
# pass naturally falls off the radar without manual cleanup.
ACCUMULATOR_TTL_SECONDS = 6 * 60 * 60


async def _record_phase_result(
    *, pass_id: str, phase: str, row: "BatchResultRow"
) -> None:
    """Persist one phase's result as a JSON blob under ``pass_id``."""
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    body = json.dumps(
        {
            "custom_id": row.custom_id,
            "content": row.content,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "cache_read_tokens": row.cache_read_tokens,
            "cache_creation_tokens": row.cache_creation_tokens,
            "error": row.error,
        }
    )
    # redis-py's typed stubs flag hash ops as returning bare values
    # rather than awaitables on AsyncRedisCluster — same workaround
    # used in ``copilot/stream_registry.py``.
    await redis.hset(_accumulator_key(pass_id), phase, body)  # type: ignore[misc]
    await redis.expire(_accumulator_key(pass_id), ACCUMULATOR_TTL_SECONDS)


async def _read_phase_results(pass_id: str) -> dict[str, dict[str, Any]]:
    """Fetch the per-phase accumulator for one pass."""
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    raw = await redis.hgetall(_accumulator_key(pass_id))  # type: ignore[misc]
    decoded: dict[str, dict[str, Any]] = {}
    for phase, body in (raw or {}).items():
        if isinstance(phase, bytes):
            phase = phase.decode("utf-8")
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        try:
            decoded[phase] = json.loads(body)
        except Exception:
            logger.warning(
                "Corrupted phase result for pass=%s phase=%s", pass_id, phase
            )
    return decoded


async def _delete_accumulator(pass_id: str) -> None:
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    await redis.delete(_accumulator_key(pass_id))


def _has_terminal_error(state: dict[str, dict[str, Any]]) -> bool:
    return any(row.get("error") for row in state.values())


REQUIRED_PHASES = ("consolidate", "recombine", "sanitize")


def _all_phases_present(state: dict[str, dict[str, Any]]) -> bool:
    return all(phase in state for phase in REQUIRED_PHASES)


# ---------------------------------------------------------------------------
# Terminal handlers
# ---------------------------------------------------------------------------


async def _finalize_errored(
    *,
    user_id: str,
    pass_id: str,
    job_id: str,
    state: dict[str, dict[str, Any]],
) -> None:
    """Mark the JobStatus row errored and clear the accumulator.

    Picks the first error string off any failed phase row for the
    status message — sufficient for the admin UI to surface; the per-
    phase logs already have the full trace.
    """
    error_phase, error_msg = _first_error(state)
    logger.warning(
        "Dream batch pass=%s errored on phase=%s: %s",
        pass_id,
        error_phase,
        error_msg,
    )
    if job_id:
        try:
            from backend.copilot.dream.job_status import mark_errored

            await mark_errored(
                kind="dream_pass",
                job_id=job_id,
                error=f"{error_phase}: {error_msg}",
            )
        except Exception:
            logger.exception("Failed to mark dream pass job %s errored", job_id)
    await _delete_accumulator(pass_id)


async def _finalize_complete(
    *,
    user_id: str,
    pass_id: str,
    job_id: str,
    state: dict[str, dict[str, Any]],
) -> None:
    """Run the apply step + cost log + JobStatus complete + cleanup.

    Apply is the only place batch results actually mutate the graph;
    everything before this is metadata pipeline.
    """
    try:
        from backend.copilot.dream.apply import apply_operations
        from backend.copilot.dream.schemas import DreamOperations

        sanitize_row = state["sanitize"]
        ops = DreamOperations.model_validate(json.loads(sanitize_row["content"]))
        stats = await apply_operations(user_id, pass_id, ops)
    except Exception as exc:
        logger.exception(
            "apply_operations crashed for batch pass=%s — marking errored",
            pass_id,
        )
        if job_id:
            try:
                from backend.copilot.dream.job_status import mark_errored

                await mark_errored(
                    kind="dream_pass",
                    job_id=job_id,
                    error=f"apply: {type(exc).__name__}: {exc}",
                )
            except Exception:
                pass
        await _delete_accumulator(pass_id)
        return

    # Per-phase cost log — applied AFTER apply succeeds so a failed
    # apply doesn't double-bill the user.
    await _log_phase_costs(user_id=user_id, pass_id=pass_id, state=state)

    if job_id:
        try:
            from backend.copilot.dream.job_status import mark_complete

            await mark_complete(
                kind="dream_pass",
                job_id=job_id,
                result={
                    "pass_id": pass_id,
                    "stats": {k: v for k, v in stats.items() if k != "snapshot"},
                },
            )
        except Exception:
            logger.exception("Failed to mark dream pass job %s complete", job_id)

    await _delete_accumulator(pass_id)


# ---------------------------------------------------------------------------
# Cost logging
# ---------------------------------------------------------------------------


async def _log_phase_costs(
    *, user_id: str, pass_id: str, state: dict[str, dict[str, Any]]
) -> None:
    """Write a PlatformCostLog row per phase, tagged anthropic_batch."""
    try:
        from backend.copilot.dream.billing import record_phase_cost
        from backend.copilot.dream.schemas import PhaseUsage
    except Exception:
        logger.exception(
            "Failed to import dream billing — skipping cost log for pass=%s",
            pass_id,
        )
        return

    for phase, row in state.items():
        try:
            usage = PhaseUsage(
                phase=phase,  # type: ignore[arg-type]
                model="claude-sonnet-4-6",
                input_tokens=int(row.get("input_tokens") or 0),
                output_tokens=int(row.get("output_tokens") or 0),
                cache_read_tokens=int(row.get("cache_read_tokens") or 0),
                cache_creation_tokens=int(row.get("cache_creation_tokens") or 0),
                cost_usd=None,  # rate card computes it with batch discount
            )
            await record_phase_cost(
                user_id=user_id,
                pass_id=pass_id,
                phase_usage=usage,
                execution_path="anthropic_batch",
            )
        except Exception:
            logger.exception(
                "Failed to log batch cost for pass=%s phase=%s", pass_id, phase
            )


def _first_error(state: dict[str, dict[str, Any]]) -> tuple[str, str]:
    for phase, row in state.items():
        err = row.get("error")
        if err:
            return phase, str(err)
    return "unknown", "no error captured"


# ---------------------------------------------------------------------------
# Registration (runs at module import)
# ---------------------------------------------------------------------------


def _register() -> None:
    try:
        from backend.executor.batch_executor import register_handler

        register_handler(NAMESPACE, handle_dream_batch_result)
    except Exception:
        logger.exception(
            "Failed to register dream batch handler — "
            "batch results will not dispatch"
        )


_register()
