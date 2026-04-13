"""Helpers for platform cost tracking on system-credential block executions."""

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, cast

from backend.blocks._base import Block, BlockSchema
from backend.copilot.token_tracking import _pending_log_tasks as _copilot_tasks
from backend.copilot.token_tracking import (
    _pending_log_tasks_lock as _copilot_tasks_lock,
)
from backend.data.execution import NodeExecutionEntry
from backend.data.model import NodeExecutionStats
from backend.data.platform_cost import PlatformCostEntry, usd_to_microdollars
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import is_system_credential
from backend.integrations.providers import ProviderName

if TYPE_CHECKING:
    from backend.data.db_manager import DatabaseManagerAsyncClient

logger = logging.getLogger(__name__)

# Provider groupings by billing model — used when the block didn't explicitly
# declare stats.provider_cost_type and we fall back to provider-name
# heuristics. Values match ProviderName enum values.
_CHARACTER_BILLED_PROVIDERS = frozenset(
    {ProviderName.D_ID.value, ProviderName.ELEVENLABS.value}
)
_WALLTIME_BILLED_PROVIDERS = frozenset(
    {
        ProviderName.FAL.value,
        ProviderName.REVID.value,
        ProviderName.REPLICATE.value,
    }
)

# Hold strong references to in-flight log tasks so the event loop doesn't
# garbage-collect them mid-execution. Tasks remove themselves on completion.
# _pending_log_tasks_lock guards all reads and writes: worker threads call
# discard() via done callbacks while drain_pending_cost_logs() iterates.
_pending_log_tasks: set[asyncio.Task] = set()
_pending_log_tasks_lock = threading.Lock()
# Per-loop semaphores: asyncio.Semaphore is not thread-safe and must not be
# shared across event loops running in different threads. Key by loop instance
# so each executor worker thread gets its own semaphore.
_log_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


def _get_log_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _log_semaphores.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(50)
        _log_semaphores[loop] = sem
    return sem


async def drain_pending_cost_logs(timeout: float = 5.0) -> None:
    """Await all in-flight cost log tasks with a timeout.

    Drains both the executor cost log tasks (_pending_log_tasks in this module,
    used for block execution cost tracking via DatabaseManagerAsyncClient) and
    the copilot cost log tasks (token_tracking._pending_log_tasks, used for
    copilot LLM turns via platform_cost_db()).

    Call this during graceful shutdown to flush pending INSERT tasks before
    the process exits. Tasks that don't complete within `timeout` seconds are
    abandoned and their failures are already logged by _safe_log.
    """
    # asyncio.wait() requires all tasks to belong to the running event loop.
    # _pending_log_tasks is shared across executor worker threads (each with
    # its own loop), so filter to only tasks owned by the current loop.
    # Acquire the lock to take a consistent snapshot (worker threads call
    # discard() via done callbacks concurrently with this iteration).
    current_loop = asyncio.get_running_loop()
    with _pending_log_tasks_lock:
        all_pending = [t for t in _pending_log_tasks if t.get_loop() is current_loop]
    if all_pending:
        logger.info("Draining %d executor cost log task(s)", len(all_pending))
        _, still_pending = await asyncio.wait(all_pending, timeout=timeout)
        if still_pending:
            logger.warning(
                "%d executor cost log task(s) did not complete within %.1fs",
                len(still_pending),
                timeout,
            )
    # Also drain copilot cost log tasks (token_tracking._pending_log_tasks)
    with _copilot_tasks_lock:
        copilot_pending = [t for t in _copilot_tasks if t.get_loop() is current_loop]
    if copilot_pending:
        logger.info("Draining %d copilot cost log task(s)", len(copilot_pending))
        _, still_pending = await asyncio.wait(copilot_pending, timeout=timeout)
        if still_pending:
            logger.warning(
                "%d copilot cost log task(s) did not complete within %.1fs",
                len(still_pending),
                timeout,
            )


def _schedule_log(
    db_client: "DatabaseManagerAsyncClient", entry: PlatformCostEntry
) -> None:
    async def _safe_log() -> None:
        async with _get_log_semaphore():
            try:
                await db_client.log_platform_cost(entry)
            except Exception:
                logger.exception(
                    "Failed to log platform cost for user=%s provider=%s block=%s",
                    entry.user_id,
                    entry.provider,
                    entry.block_name,
                )

    task = asyncio.create_task(_safe_log())
    with _pending_log_tasks_lock:
        _pending_log_tasks.add(task)

    def _remove(t: asyncio.Task) -> None:
        with _pending_log_tasks_lock:
            _pending_log_tasks.discard(t)

    task.add_done_callback(_remove)


def _extract_model_name(raw: str | dict | None) -> str | None:
    """Return a string model name from a block input field, or None.

    Handles str (returned as-is), dict (e.g. an enum wrapper, skipped), and
    None (no model field). Unexpected types are coerced to str as a fallback.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return None
    return str(raw)


def resolve_tracking(
    provider: str,
    stats: NodeExecutionStats,
    input_data: dict[str, Any],
) -> tuple[str, float]:
    """Return (tracking_type, tracking_amount) based on provider billing model.

    Preference order:
    1. Block-declared: if the block set `provider_cost_type` on its stats,
       honor it directly (paired with `provider_cost` as the amount).
    2. Heuristic fallback: infer from `provider_cost`/token counts, then
       from provider name for per-character / per-second billing.
    """
    # 1. Block explicitly declared its cost type (only when an amount is present)
    if stats.provider_cost_type and stats.provider_cost is not None:
        return stats.provider_cost_type, max(0.0, stats.provider_cost)

    # 2. Provider returned actual USD cost (OpenRouter, Exa)
    if stats.provider_cost is not None:
        return "cost_usd", max(0.0, stats.provider_cost)

    # 3. LLM providers: track by tokens
    if stats.input_token_count or stats.output_token_count:
        return "tokens", float(
            (stats.input_token_count or 0) + (stats.output_token_count or 0)
        )

    # 4. Provider-specific billing heuristics

    # TTS: billed per character of input text
    if provider == ProviderName.UNREAL_SPEECH.value:
        text = input_data.get("text", "")
        return "characters", float(len(text)) if isinstance(text, str) else 0.0

    # D-ID + ElevenLabs voice: billed per character of script
    if provider in _CHARACTER_BILLED_PROVIDERS:
        text = (
            input_data.get("script_input", "")
            or input_data.get("text", "")
            or input_data.get("script", "")  # VideoNarrationBlock uses `script`
        )
        return "characters", float(len(text)) if isinstance(text, str) else 0.0

    # E2B: billed per second of sandbox time
    if provider == ProviderName.E2B.value:
        return "sandbox_seconds", round(stats.walltime, 3) if stats.walltime else 0.0

    # Video/image gen: walltime includes queue + generation + polling
    if provider in _WALLTIME_BILLED_PROVIDERS:
        return "walltime_seconds", round(stats.walltime, 3) if stats.walltime else 0.0

    # Per-request: Google Maps, Ideogram, Nvidia, Apollo, etc.
    # All billed per API call - count 1 per block execution.
    return "per_run", 1.0


async def log_system_credential_cost(
    node_exec: NodeExecutionEntry,
    block: Block,
    stats: NodeExecutionStats,
    db_client: "DatabaseManagerAsyncClient",
) -> None:
    """Check if a system credential was used and log the platform cost.

    Routes through DatabaseManagerAsyncClient so the write goes via the
    message-passing DB service rather than calling Prisma directly (which
    is not connected in the executor process).

    Logs only the first matching system credential field (one log per
    execution). Any unexpected error is caught and logged — cost logging
    is strictly best-effort and must never disrupt block execution.

    Note: costMicrodollars is left null for providers that don't return
    a USD cost. The credit_cost in metadata captures our internal credit
    charge as a proxy.
    """
    try:
        if node_exec.execution_context.dry_run:
            return

        input_data = node_exec.inputs
        input_model = cast(type[BlockSchema], block.input_schema)

        for field_name in input_model.get_credentials_fields():
            cred_data = input_data.get(field_name)
            if not cred_data or not isinstance(cred_data, dict):
                continue
            cred_id = cred_data.get("id", "")
            if not cred_id or not is_system_credential(cred_id):
                continue

            model_name = _extract_model_name(input_data.get("model"))

            credit_cost, _ = block_usage_cost(block=block, input_data=input_data)

            provider_name = cred_data.get("provider", "unknown")
            tracking_type, tracking_amount = resolve_tracking(
                provider=provider_name,
                stats=stats,
                input_data=input_data,
            )

            # Only treat provider_cost as USD when the tracking type says so.
            # For other types (items, characters, per_run, ...) the
            # provider_cost field holds the raw amount, not a dollar value.
            # Use tracking_amount (the normalized value from resolve_tracking)
            # rather than raw stats.provider_cost to avoid unit mismatches.
            cost_microdollars = None
            if tracking_type == "cost_usd":
                cost_microdollars = usd_to_microdollars(tracking_amount)

            meta: dict[str, Any] = {
                "tracking_type": tracking_type,
                "tracking_amount": tracking_amount,
            }
            if credit_cost is not None:
                meta["credit_cost"] = credit_cost
            if stats.provider_cost is not None:
                # Use 'provider_cost_raw' — the value's unit varies by tracking
                # type (USD for cost_usd, count for items/characters/per_run, etc.)
                meta["provider_cost_raw"] = stats.provider_cost

            _schedule_log(
                db_client,
                PlatformCostEntry(
                    user_id=node_exec.user_id,
                    graph_exec_id=node_exec.graph_exec_id,
                    node_exec_id=node_exec.node_exec_id,
                    graph_id=node_exec.graph_id,
                    node_id=node_exec.node_id,
                    block_id=node_exec.block_id,
                    block_name=block.name,
                    provider=provider_name,
                    credential_id=cred_id,
                    cost_microdollars=cost_microdollars,
                    input_tokens=stats.input_token_count,
                    output_tokens=stats.output_token_count,
                    cache_read_tokens=stats.cache_read_token_count or None,
                    cache_creation_tokens=stats.cache_creation_token_count or None,
                    data_size=stats.output_size if stats.output_size > 0 else None,
                    duration=stats.walltime if stats.walltime > 0 else None,
                    model=model_name,
                    tracking_type=tracking_type,
                    tracking_amount=tracking_amount,
                    metadata=meta,
                ),
            )
            return  # One log per execution is enough
    except Exception:
        logger.exception("log_system_credential_cost failed unexpectedly")
