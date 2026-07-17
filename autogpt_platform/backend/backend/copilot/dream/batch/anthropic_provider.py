"""Anthropic Message Batches adapter — IMPLEMENTATION STUB.

The real adapter wraps ``anthropic.AsyncAnthropic().messages.batches``.
That client lives behind the ``anthropic`` package; the dream pipeline
ships with the package as a dependency (used by the SDK chat path)
so no extra install is needed.

Stub today, real impl in a follow-up PR. Filling this in is mostly
mechanical:

1. ``submit`` — call ``client.messages.batches.create(requests=[
   {"custom_id": r.custom_id, "params": {...}} for r in requests])``,
   map the returned ``Batch`` object onto ``BatchSubmission``.
2. ``poll`` — call ``client.messages.batches.retrieve(batch_id)``,
   normalize ``processing_status`` (``in_progress`` /
   ``canceling`` / ``ended``) onto ``BatchStatus``.
3. ``download_results`` — call
   ``client.messages.batches.results(batch_id)``; the body is JSONL
   with one line per request. Parse each line, extract
   ``result.message.content`` + ``result.message.usage``, build a
   ``BatchResult``. ``result.type == "errored"`` → set ``error``.

The Pydantic models in ``models.py`` are deliberately narrower than
Anthropic's full response shape so the adapter has freedom to drop
fields we don't use (e.g. ``request_counts``).

The ``cost_usd`` field is computed via ``model_pricing.compute_cost_usd``
with ``execution_path="anthropic_batch"`` so the 50% discount is
applied. Anthropic doesn't return a ``cost`` field; computing it
from tokens against our rate card is the contract.
"""

from __future__ import annotations

from typing import Sequence

from .models import BatchRequest, BatchResult, BatchSubmission


class AnthropicBatchProvider:
    """Stub placeholder. See module docstring for the implementation
    plan. Calling any method raises ``NotImplementedError`` with a
    pointer to ``p0-spec.md §P0.1``."""

    name = "anthropic"

    async def submit(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
        raise NotImplementedError(
            "AnthropicBatchProvider.submit not yet implemented — "
            "see dream/p0-spec.md §P0.1 and "
            "dream/batch/anthropic_provider.py module docstring."
        )

    async def poll(self, submission: BatchSubmission) -> BatchSubmission:
        raise NotImplementedError("AnthropicBatchProvider.poll not yet implemented.")

    async def download_results(self, submission: BatchSubmission) -> list[BatchResult]:
        raise NotImplementedError(
            "AnthropicBatchProvider.download_results not yet implemented."
        )
