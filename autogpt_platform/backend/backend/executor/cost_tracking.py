"""Helpers for platform cost tracking on system-credential block executions."""

import asyncio
import logging
from typing import Any, cast

from backend.blocks._base import BlockSchema
from backend.data.model import NodeExecutionStats
from backend.data.platform_cost import PlatformCostEntry, log_platform_cost_safe
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import is_system_credential

logger = logging.getLogger(__name__)


def resolve_tracking(
    provider: str,
    block_name: str,
    stats: NodeExecutionStats,
    input_data: dict[str, Any],
) -> tuple[str, float]:
    """Return (tracking_type, tracking_amount) based on provider billing model."""
    # 1. Provider returned actual USD cost (OpenRouter, Exa)
    if stats.provider_cost is not None:
        return "cost_usd", stats.provider_cost

    # 2. LLM providers: track by tokens
    if stats.input_token_count or stats.output_token_count:
        return "tokens", float(
            (stats.input_token_count or 0) + (stats.output_token_count or 0)
        )

    # 3. Provider-specific billing models

    # TTS: billed per character of input text
    if provider == "unreal_speech":
        text = input_data.get("text", "")
        return "characters", float(len(text)) if isinstance(text, str) else 0.0

    # D-ID + ElevenLabs voice: billed per character of script
    if provider in ("d_id", "elevenlabs"):
        text = input_data.get("script_input", "") or input_data.get("text", "")
        return "characters", float(len(text)) if isinstance(text, str) else 0.0

    # E2B: billed per second of sandbox time
    if provider == "e2b":
        return "sandbox_seconds", round(stats.walltime, 3) if stats.walltime else 0.0

    # Video/image gen: walltime includes queue + generation + polling
    if provider in ("fal", "revid", "replicate"):
        return "walltime_seconds", round(stats.walltime, 3) if stats.walltime else 0.0

    # Per-request: Google Maps, Ideogram, Nvidia, Apollo, etc.
    # All billed per API call - count 1 per block execution.
    # output_size captured separately for volume estimation.
    return "per_run", 1.0


async def log_system_credential_cost(
    node_exec: Any,
    block: Any,
    stats: NodeExecutionStats,
) -> None:
    """Check if a system credential was used and log the platform cost.

    Note: costMicrodollars is left null for now. To populate it, we would
    need per-token pricing tables or extract cost from provider responses
    (e.g. OpenRouter returns a cost field). The credit_cost in metadata
    captures our internal credit charge as a proxy.
    """
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

        model_name = input_data.get("model")
        if model_name is not None and not isinstance(model_name, str):
            model_name = str(model_name) if not isinstance(model_name, dict) else None

        credit_cost, _ = block_usage_cost(block=block, input_data=input_data)

        # Convert provider_cost (USD) to microdollars if available
        cost_microdollars = None
        if stats.provider_cost is not None:
            cost_microdollars = round(stats.provider_cost * 1_000_000)

        provider_name = cred_data.get("provider", "unknown")
        tracking_type, tracking_amount = resolve_tracking(
            provider=provider_name,
            block_name=block.name,
            stats=stats,
            input_data=input_data,
        )

        meta: dict[str, Any] = {
            "tracking_type": tracking_type,
            "tracking_amount": tracking_amount,
        }
        if credit_cost:
            meta["credit_cost"] = credit_cost
        if stats.provider_cost is not None:
            meta["provider_cost_usd"] = stats.provider_cost

        asyncio.create_task(
            log_platform_cost_safe(
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
                    input_tokens=stats.input_token_count or None,
                    output_tokens=stats.output_token_count or None,
                    data_size=stats.output_size or None,
                    duration=stats.walltime or None,
                    model=model_name,
                    tracking_type=tracking_type,
                    metadata=meta or None,
                )
            )
        )
        return  # One log per execution is enough
