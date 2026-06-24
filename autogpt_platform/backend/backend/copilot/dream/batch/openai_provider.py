"""OpenAI Batch API adapter — IMPLEMENTATION STUB.

The real adapter wraps ``openai.AsyncOpenAI().batches`` + ``.files``.
OpenAI's batch flow is two-step:

1. Upload a JSONL file via ``client.files.create(purpose="batch")``.
   Each line is one ``ChatCompletion`` request with a ``custom_id``.
2. Create a batch via ``client.batches.create(input_file_id=...,
   endpoint="/v1/chat/completions", completion_window="24h")``.

Polling/downloading mirror the Anthropic flow but the JSONL line
format differs — see https://platform.openai.com/docs/guides/batch.

Stub today, real impl in a follow-up PR. Filling this in is mostly
mechanical:

1. ``submit`` — write the requests to an in-memory buffer, upload
   via ``files.create``, then ``batches.create``.
2. ``poll`` — ``batches.retrieve(batch_id)``; normalize
   ``status`` (``validating`` / ``in_progress`` / ``finalizing`` /
   ``completed`` / ``failed``) onto ``BatchStatus``.
3. ``download_results`` — download ``output_file_id`` via
   ``files.content(...)``, parse JSONL, extract
   ``response.body.choices[0].message.content`` +
   ``response.body.usage``. Track ``error_file_id`` separately for
   per-request errors.

The ``cost_usd`` field comes from ``model_pricing.compute_cost_usd``
with ``execution_path="openai_batch"`` (50% discount applied).
OpenAI's batch responses don't carry a cost field either.
"""

from __future__ import annotations

from typing import Sequence

from .models import BatchRequest, BatchResult, BatchSubmission


class OpenAIBatchProvider:
    """Stub placeholder. See module docstring for the implementation
    plan. Calling any method raises ``NotImplementedError`` with a
    pointer to ``p0-spec.md §P0.1``."""

    name = "openai"

    async def submit(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
        raise NotImplementedError(
            "OpenAIBatchProvider.submit not yet implemented — "
            "see dream/p0-spec.md §P0.1 and "
            "dream/batch/openai_provider.py module docstring."
        )

    async def poll(self, submission: BatchSubmission) -> BatchSubmission:
        raise NotImplementedError("OpenAIBatchProvider.poll not yet implemented.")

    async def download_results(self, submission: BatchSubmission) -> list[BatchResult]:
        raise NotImplementedError(
            "OpenAIBatchProvider.download_results not yet implemented."
        )
