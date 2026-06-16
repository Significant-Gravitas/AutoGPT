"""BatchProvider Protocol + a Null implementation.

Every concrete batch provider (Anthropic, OpenAI) implements this
Protocol. The Null provider always reports submission failure with a
documented error so the orchestrator falls back to sync_baseline. The
dream pipeline can wire BatchProvider in today and pick up
real-Anthropic / real-OpenAI when those adapters land.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol, Sequence

from .models import BatchRequest, BatchResult, BatchStatus, BatchSubmission

logger = logging.getLogger(__name__)


class BatchProvider(Protocol):
    """What a concrete batch backend (Anthropic / OpenAI) must implement.

    Lifecycle:
      1. ``submit(requests)`` → returns a ``BatchSubmission`` whose
         ``provider_batch_id`` is the lookup key.
      2. ``poll(submission)`` → returns the same submission with an
         updated ``status``. Caller decides when to poll again.
      3. ``download_results(submission)`` → returns the per-request
         responses. Only safe to call when status is ``ENDED``.

    Cancellation, retries, and exponential-backoff polling cadence
    are the *caller's* responsibility (lives in ``poller.py`` once
    that module is filled in). The provider only owns the API surface.
    """

    name: str

    async def submit(self, requests: Sequence[BatchRequest]) -> BatchSubmission: ...

    async def poll(self, submission: BatchSubmission) -> BatchSubmission: ...

    async def download_results(
        self, submission: BatchSubmission
    ) -> list[BatchResult]: ...


class NullBatchProvider:
    """Always-fails BatchProvider.

    Returned by ``factory.get_batch_provider`` when neither Anthropic
    nor OpenAI is configured (no API key, or batch flag off). Surfaces
    a structured "no batch backend" error so the orchestrator's batch
    path can short-circuit to sync_baseline cleanly instead of crashing
    on a missing adapter.

    Submitting against this provider yields a synthetic ``FAILED``
    submission whose ``provider_batch_id`` is empty; polling and
    downloading return the same status and an empty result list. The
    orchestrator should never reach the download step against a Null
    provider, but defending against it is a one-liner so we do.
    """

    name = "null"

    async def submit(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
        logger.info(
            "NullBatchProvider.submit() called with %d requests — no "
            "batch backend configured; caller should fall back to "
            "sync_baseline.",
            len(requests),
        )
        return BatchSubmission(
            provider_batch_id="",
            provider=self.name,
            submitted_at=datetime.now(timezone.utc),
            status=BatchStatus.FAILED,
        )

    async def poll(self, submission: BatchSubmission) -> BatchSubmission:
        return submission

    async def download_results(self, submission: BatchSubmission) -> list[BatchResult]:
        return []
