"""Submit one dream phase to Anthropic batch + enqueue with BatchExecutor.

Both the orchestrator (kicking off phase 1) and the dream
``batch_callbacks`` (chaining phases 2 → 3 as results arrive) need to
submit identically: build the per-phase prompt with whatever upstream
phase outputs we have, force tool_use structured output against the
phase's Pydantic schema, then enqueue the submission so the
BatchExecutor polls it. This module owns that flow.

Splitting it out:
  * Keeps the orchestrator focused on routing / cost accounting and
    not duplicating Anthropic message-shape construction.
  * Keeps the callback module focused on lifecycle (which phase
    landed, what's next, mark complete vs errored) rather than
    SDK plumbing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from backend.executor.batch_executor import (
    INITIAL_POLL_DELAY_SECONDS,
    PendingEntry,
    enqueue_pending,
)
from backend.util.llm.providers import (
    BatchSubmissionRef,
    ProviderResponse,
    call_provider,
    cancel_batch,
)
from backend.util.llm.tool_use import force_tool_choice, pydantic_to_anthropic_tool

from .fetch import DreamInput
from .locks import read_dream_lock_token
from .prompts import (
    build_consolidate_prompt,
    build_recombine_prompt,
    build_sanitize_prompt,
)
from .schemas import ConsolidationOutput, DreamOperations, RecombinationOutput

if TYPE_CHECKING:
    from backend.copilot.config import ChatConfig

logger = logging.getLogger(__name__)

DreamPhase = Literal["consolidate", "recombine", "sanitize"]


# Tool name the model must emit for each phase. Forced ``tool_choice``
# constrains output to exactly one tool_use block whose ``input`` matches
# the corresponding Pydantic schema — no JSON-parse-prose failures.
PHASE_TOOL_NAMES: dict[DreamPhase, str] = {
    "consolidate": "emit_consolidation",
    "recombine": "emit_recombination",
    "sanitize": "emit_dream_operations",
}


PHASE_RESPONSE_MODELS: dict[DreamPhase, type[BaseModel]] = {
    "consolidate": ConsolidationOutput,
    "recombine": RecombinationOutput,
    "sanitize": DreamOperations,
}


# Max output tokens per phase (mirrors the sync orchestrator's values).
PHASE_MAX_TOKENS: dict[DreamPhase, int] = {
    "consolidate": 4096,
    "recombine": 16384,
    "sanitize": 16384,
}


PHASE_TEMPERATURES: dict[DreamPhase, float] = {
    "consolidate": 0.2,
    "recombine": 0.9,
    "sanitize": 0.0,
}


PHASE_DESCRIPTIONS: dict[DreamPhase, str] = {
    "consolidate": (
        "Emit the consolidated facts produced from the user's recent "
        "episodes + active facts."
    ),
    "recombine": (
        "Emit novel recombination proposals drawn from the " "consolidated facts."
    ),
    "sanitize": (
        "Emit the gated final dream operations (writes / proposals / "
        "demotions / entity invalidations / summary_for_user)."
    ),
}


def _to_native_anthropic_model(model: str) -> str:
    """``anthropic/claude-opus-4.7`` → ``claude-opus-4-7``.

    The batch path always submits to Anthropic's native Batches API
    (direct key), regardless of the chat transport, so strip the
    OpenRouter vendor prefix AND convert dots to hyphens — native
    Anthropic rejects both the prefix and the dot-separated version,
    and the dream rate card only carries the hyphenated id.
    """
    if "/" in model:
        model = model.split("/", 1)[1]
    return model.replace(".", "-")


def phase_models_for_config(config: "ChatConfig") -> dict[str, str]:
    """Per-phase model map (native-Anthropic form) for the batch path.

    Mirrors the sync orchestrator's per-phase choice: the standard tier
    for consolidate/sanitize, the advanced tier (opus + extended
    thinking) for recombine. Built once at submit and threaded through
    the batch payload so the model used to *submit* a phase is the exact
    model used to *price* it — no single-model fan-out across phases.
    """
    return {
        "consolidate": _to_native_anthropic_model(config.fast_standard_model),
        "recombine": _to_native_anthropic_model(config.fast_advanced_model),
        "sanitize": _to_native_anthropic_model(config.fast_standard_model),
    }


async def submit_phase(
    *,
    user_id: str,
    pass_id: str,
    job_id: str,
    phase: DreamPhase,
    phase_models: dict[str, str],
    api_key: str,
    input_bundle: DreamInput,
    consolidated_json: str | None = None,
    recombined_json: str | None = None,
) -> BatchSubmissionRef:
    """Submit one dream phase as a single-request Anthropic batch.

    The custom_id encodes ``{pass_id}:{phase}`` so the BatchExecutor's
    dispatch + dream callbacks can correlate the result back. Returns
    the ``BatchSubmissionRef`` so the caller can persist the batch_id
    on the JobStatus row.

    The pending-queue entry's ``payload`` carries everything the
    callback needs to chain to the next phase: user_id, pass_id,
    job_id, and the per-phase model map so each phase is submitted —
    and later priced — with its own model.
    """
    model = phase_models[phase]
    messages = _build_phase_messages(
        phase=phase,
        input_bundle=input_bundle,
        consolidated_json=consolidated_json,
        recombined_json=recombined_json,
    )
    tool_name = PHASE_TOOL_NAMES[phase]
    tools = [
        pydantic_to_anthropic_tool(
            PHASE_RESPONSE_MODELS[phase],
            tool_name=tool_name,
            description=PHASE_DESCRIPTIONS[phase],
        )
    ]
    # Anthropic's custom_id must match ``^[a-zA-Z0-9_-]{1,64}$`` —
    # colons aren't allowed. Underscore separator keeps the
    # ``pass_id_phase`` structure parseable by the dream callback
    # while staying within the provider's alphabet.
    custom_id = f"{pass_id}_{phase}"

    submission = await call_provider(
        provider="anthropic",
        model=model,
        api_key=api_key,
        messages=messages,
        max_tokens=PHASE_MAX_TOKENS[phase],
        temperature=PHASE_TEMPERATURES[phase],
        tools=tools,
        tool_choice=force_tool_choice(tool_name),
        execution_mode="batch",
        custom_id=custom_id,
    )
    if isinstance(submission, ProviderResponse):
        raise RuntimeError(
            "submit_phase requested execution_mode='batch' but "
            "call_provider returned ProviderResponse — Anthropic batch "
            "may be unsupported for the configured model."
        )

    entry = PendingEntry(
        provider="anthropic",
        provider_batch_id=submission.provider_batch_id,
        callback_namespace="dream_pass",
        submitted_at=submission.submitted_at,
        next_poll_at=submission.submitted_at,
        poll_delay_seconds=INITIAL_POLL_DELAY_SECONDS,
        payload={
            "user_id": user_id,
            "pass_id": pass_id,
            "job_id": job_id,
            "phase": phase,
            "phase_models": phase_models,
            "custom_ids": [custom_id],
            # The phase mapping is what dream_callbacks uses to route
            # the result row back to a phase label. With one custom_id
            # per batch we could just look it up by phase, but keeping
            # the explicit mapping lets us group requests into one
            # batch in the future without touching the callback.
            "phase_for_custom_id": {custom_id: phase},
        },
    )
    try:
        await enqueue_pending(entry)
    except Exception:
        # The provider already accepted (and will bill for) this batch, but
        # the BatchExecutor now has no pending entry to poll it — left as-is
        # it would run to completion with no callback, orphaning the spend
        # and stalling the pass. Best-effort cancel the provider batch before
        # surfacing the failure so the orchestrator releases the lock + marks
        # the job failed without leaving a paid, un-pollable batch behind.
        logger.exception(
            "Dream pass=%s phase=%s: enqueue failed after batch=%s submitted — "
            "cancelling orphaned provider batch",
            pass_id,
            phase,
            submission.provider_batch_id,
        )
        await cancel_batch(
            provider="anthropic",
            provider_batch_id=submission.provider_batch_id,
            api_key=api_key,
        )
        raise
    logger.info(
        "Dream pass=%s submitted %s batch=%s",
        pass_id,
        phase,
        submission.provider_batch_id,
    )
    return submission


def _build_phase_messages(
    *,
    phase: DreamPhase,
    input_bundle: DreamInput,
    consolidated_json: str | None,
    recombined_json: str | None,
) -> list[dict[str, str]]:
    """Build the OpenAI-format message list for one phase.

    Reuses the existing prompt builders so batch and sync paths
    produce identical inputs — only the transport (batch vs sync)
    and the structured-output enforcement (tool_use vs response_format)
    differ.
    """
    if phase == "consolidate":
        return build_consolidate_prompt(input_bundle)
    if phase == "recombine":
        if consolidated_json is None:
            raise ValueError(
                "recombine phase requires consolidated_json from the "
                "previous phase's result."
            )
        return build_recombine_prompt(input_bundle, consolidated_json)
    if phase == "sanitize":
        if consolidated_json is None or recombined_json is None:
            raise ValueError(
                "sanitize phase requires both consolidated_json and " "recombined_json."
            )
        return build_sanitize_prompt(input_bundle, consolidated_json, recombined_json)
    raise ValueError(f"Unknown dream phase: {phase}")


# ---------------------------------------------------------------------------
# DreamInput persistence across batch result callbacks
# ---------------------------------------------------------------------------


def input_bundle_key(pass_id: str) -> str:
    return f"dream:batch:input:{pass_id}"


# 24h TTL matches Anthropic's batch SLA — if a batch hasn't completed
# in 24h we've already timed out via BatchExecutor's
# ``MAX_BATCH_LIFETIME_SECONDS`` ceiling.
INPUT_TTL_SECONDS = 24 * 60 * 60


async def persist_input_bundle(
    pass_id: str,
    input_bundle: DreamInput,
    *,
    lock_token: str | None = None,
) -> None:
    """Serialize ``DreamInput`` to Redis so callbacks can rebuild prompts.

    DreamInput carries lists of dataclasses; we round-trip through
    ``json`` rather than pickling so the wire format stays portable +
    debuggable (`docker exec redis ... HGET ...` works).

    The stored body also carries the dream lock's ownership token
    (``lock_token``): the orchestrator persists the bundle while it still
    holds the lock, and the batch callback — hours later, in a different
    process — needs that token for the compare-and-delete release in
    ``release_dream_lock``. Riding on this key avoids a second per-pass
    key and keeps the batch state single-key for cluster mode.

    The orchestrator passes its lock handle's own ``lock_token``
    explicitly — reading the live lock key here instead could capture a
    *newer* pass's token (lock expired and was re-acquired between this
    pass's acquire and the persist), letting this pass's callback release
    a lock it doesn't own hours later. Only when no token is supplied
    (eval harness calling the batch path without a lock) do we fall back
    to the live key.
    """
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    payload = _input_bundle_to_dict(input_bundle)
    if lock_token is None:
        lock_token = await read_dream_lock_token(input_bundle.user_id)
    if lock_token is not None:
        payload["lock_token"] = lock_token
    await redis.set(
        input_bundle_key(pass_id), json.dumps(payload), ex=INPUT_TTL_SECONDS
    )


async def read_lock_token(pass_id: str) -> str | None:
    """Dream-lock ownership token persisted alongside the input bundle.

    None when the bundle is gone (TTL expired or already cleaned up), is
    corrupted, or was written while no lock was held — the caller then
    leaves the lock to its TTL rather than risking a blind delete of a
    newer pass's lock.
    """
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    raw = await redis.get(input_bundle_key(pass_id))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        token = json.loads(raw).get("lock_token")
    except Exception:
        logger.exception(
            "Corrupted DreamInput in Redis for pass=%s — no lock token", pass_id
        )
        return None
    return token if isinstance(token, str) else None


async def read_input_bundle(pass_id: str) -> DreamInput | None:
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    raw = await redis.get(input_bundle_key(pass_id))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return _dict_to_input_bundle(json.loads(raw))
    except Exception:
        logger.exception("Corrupted DreamInput in Redis for pass=%s — bailing", pass_id)
        return None


async def delete_input_bundle(pass_id: str) -> None:
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    await redis.delete(input_bundle_key(pass_id))


def _input_bundle_to_dict(input_bundle: DreamInput) -> dict:
    return {
        "user_id": input_bundle.user_id,
        "group_id": input_bundle.group_id,
        "window_start": input_bundle.window_start.isoformat(),
        "window_end": input_bundle.window_end.isoformat(),
        "episodes": [
            {
                "uuid": e.uuid,
                "name": e.name,
                "content": e.content,
                "source_description": e.source_description,
                "valid_at": e.valid_at,
                "created_at": e.created_at,
            }
            for e in input_bundle.episodes
        ],
        "facts": [
            {
                "uuid": f.uuid,
                "source": f.source,
                "target": f.target,
                "name": f.name,
                "fact": f.fact,
                "scope": f.scope,
                "confidence": f.confidence,
                "status": f.status,
                "created_at": f.created_at,
            }
            for f in input_bundle.facts
        ],
        "recent_sessions": [
            {
                "session_id": s.session_id,
                "title": s.title,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "body": s.body,
            }
            for s in input_bundle.recent_sessions
        ],
        "known_fact_uuids": list(input_bundle.known_fact_uuids),
        "known_episode_uuids": list(input_bundle.known_episode_uuids),
    }


def _dict_to_input_bundle(data: dict) -> DreamInput:
    from .fetch import EpisodeRow, FactRow, SessionRow

    return DreamInput(
        user_id=data["user_id"],
        group_id=data["group_id"],
        window_start=datetime.fromisoformat(data["window_start"]),
        window_end=datetime.fromisoformat(data["window_end"]),
        episodes=[EpisodeRow(**e) for e in data.get("episodes") or []],
        facts=[FactRow(**f) for f in data.get("facts") or []],
        recent_sessions=[
            SessionRow(
                session_id=s["session_id"],
                title=s.get("title"),
                created_at=(
                    datetime.fromisoformat(s["created_at"])
                    if s.get("created_at")
                    else None
                ),
                body=s.get("body") or "",
            )
            for s in data.get("recent_sessions") or []
        ],
        known_fact_uuids=set(data.get("known_fact_uuids") or []),
        known_episode_uuids=set(data.get("known_episode_uuids") or []),
    )


def _dummy_now() -> datetime:
    """Stub kept for symmetry with Step 4b; never called."""
    return datetime.now(timezone.utc)
