"""Provider-agnostic typed shapes for batch dream passes.

Both Anthropic and OpenAI expose a similar API surface:
  1. Submit N message-completion requests in one HTTP call.
  2. Poll until the batch is processing → ended.
  3. Download JSONL results, route per-request by ``custom_id``.

These models normalize that surface so the orchestrator never knows
which provider it's talking to. Per-provider quirks (Anthropic's
``ended_at`` vs OpenAI's ``completed_at``, beta header requirements,
JSONL line format) live in the per-provider adapter.

The schema is intentionally narrower than either provider's full
API — we only model what the dream pipeline uses, so neither provider
has to be feature-complete to wire in.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BatchStatus(str, Enum):
    """Lifecycle states a submitted batch can be in.

    Both providers have richer status enums (Anthropic adds
    ``canceling``, ``expired``; OpenAI adds ``finalizing``,
    ``cancelling``). Normalize to these four because the dream
    pipeline only cares about "is this done yet" + "should I retry".
    """

    PENDING = "pending"
    """Submitted; provider hasn't started processing."""
    PROCESSING = "processing"
    """Provider is generating responses."""
    ENDED = "ended"
    """Finished — results ready to download."""
    FAILED = "failed"
    """Provider returned an unrecoverable error. Surfaces as ``error``."""


class BatchRequest(BaseModel):
    """One LLM call inside a submitted batch.

    ``custom_id`` is opaque to the provider — we use it to route the
    response back to the correct dream pass + phase. Convention:
    ``"{pass_id}:{phase}"``. The dream pipeline only ever submits 3
    requests per pass (one per phase), so collisions are impossible.
    """

    custom_id: str = Field(description="Routing key the orchestrator uses.")
    model: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    response_format: dict[str, Any] | None = None
    max_output_tokens: int = 4096
    temperature: float = 0.0


class BatchResult(BaseModel):
    """One LLM response inside a downloaded batch result.

    Mirrors ``StructuredCompletion`` from ``dream/llm.py`` but
    deserialized from a batch JSONL line rather than a real-time
    completion. The orchestrator hands ``content`` to the same parser
    that handles sync responses (``_strip_json_code_fence`` +
    ``_extract_first_json_object``), so phase-2/3 lenient parsing
    keeps working on batch results.
    """

    custom_id: str
    content: str
    """Raw response body — JSON string ready for the phase parser."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None
    """Per-request error (e.g. moderation rejection); does NOT fail
    the whole batch. Surface so the orchestrator can record it as
    a phase failure and supersede the dream pass."""


class BatchSubmission(BaseModel):
    """Provider's acknowledgement of a submission.

    ``provider_batch_id`` is what the poller uses to look up status
    later — Anthropic returns a UUID, OpenAI returns ``"batch_..."``.
    Both are opaque strings to us.
    """

    provider_batch_id: str
    provider: str
    """``"anthropic"`` | ``"openai"`` — the same string used as the
    ``provider`` tag on ``PlatformCostLog`` rows."""
    submitted_at: datetime
    status: BatchStatus = BatchStatus.PENDING
