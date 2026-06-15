"""Batch scaffolding tests — pins the Protocol shape + factory routing.

Real-provider tests land when the Anthropic / OpenAI adapters are
filled in (currently NotImplementedError stubs). For now we only
need to assert:
  * BatchRequest / BatchResult round-trip via Pydantic.
  * NullBatchProvider's surface matches the BatchProvider Protocol.
  * The factory routes to the right provider per ExecutionPath.
  * Stub providers raise NotImplementedError with a spec pointer
    (so anyone calling them gets a clear next step).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from .anthropic_provider import AnthropicBatchProvider
from .factory import USE_STUB_BATCH_PROVIDERS, get_batch_provider
from .models import BatchRequest, BatchResult, BatchStatus, BatchSubmission
from .openai_provider import OpenAIBatchProvider
from .provider import BatchProvider, NullBatchProvider

# ---------------------------------------------------------------------------
# Model round-trips
# ---------------------------------------------------------------------------


def test_batch_request_round_trips_through_pydantic_dump():
    req = BatchRequest(
        custom_id="pass-1:consolidate",
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
    )
    revived = BatchRequest.model_validate(req.model_dump())
    assert revived == req


def test_batch_result_defaults_cost_to_none_when_provider_silent():
    """``cost_usd=None`` is the documented "provider didn't return
    cost" sentinel that the orchestrator's rate-card fallback keys off."""
    r = BatchResult(custom_id="x", content="{}", model="m")
    assert r.cost_usd is None
    assert r.error is None


def test_batch_submission_default_status_is_pending():
    sub = BatchSubmission(
        provider_batch_id="b1",
        provider="anthropic",
        submitted_at=datetime.now(timezone.utc),
    )
    assert sub.status == BatchStatus.PENDING


# ---------------------------------------------------------------------------
# NullBatchProvider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_null_provider_submit_reports_failed_status_with_empty_batch_id():
    """The Null provider's contract: never succeed, fail in a way the
    orchestrator can read structurally (no exception)."""
    provider = NullBatchProvider()
    sub = await provider.submit([BatchRequest(custom_id="x", model="m")])
    assert sub.status == BatchStatus.FAILED
    assert sub.provider_batch_id == ""
    assert sub.provider == "null"


@pytest.mark.asyncio
async def test_null_provider_download_returns_empty_list():
    provider = NullBatchProvider()
    sub = BatchSubmission(
        provider_batch_id="",
        provider="null",
        submitted_at=datetime.now(timezone.utc),
        status=BatchStatus.FAILED,
    )
    assert await provider.download_results(sub) == []


def test_null_provider_satisfies_batch_provider_protocol():
    """Protocol structural check — keeps the contract honest when
    we add methods to BatchProvider later."""
    provider: BatchProvider = NullBatchProvider()
    assert provider.name == "null"


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------


def test_factory_returns_null_for_sync_baseline():
    """sync_baseline never uses batch — orchestrator gates upstream,
    but the factory still has to handle the path without raising."""
    provider = get_batch_provider("sync_baseline")
    assert isinstance(provider, NullBatchProvider)


def test_use_stub_batch_providers_flag_is_off():
    """Step 5 of the rollout flipped this off: the orchestrator's
    Anthropic batch path now goes directly through
    ``backend/util/llm/providers.call_provider(execution_mode="batch")``.
    The legacy factory + stub providers are kept around only for
    callers that still want the Protocol-shaped abstraction (P-0.1
    callers and the future OpenAI batch path)."""
    assert USE_STUB_BATCH_PROVIDERS is False


def test_factory_returns_anthropic_provider_when_unstubbed():
    """With the stub flag off the factory hands back the real
    AnthropicBatchProvider for ``anthropic_batch`` paths. Today this
    is unused by the orchestrator (which calls ``call_provider``
    directly) but kept for adapter symmetry once OpenAI batch lands."""
    provider = get_batch_provider("anthropic_batch")
    # Specifically NOT NullBatchProvider — the stub-gated test above
    # would catch a regression flipping the flag back to True.
    assert not isinstance(provider, NullBatchProvider)


def test_factory_returns_openai_provider_when_unstubbed():
    """Same contract for OpenAI."""
    provider = get_batch_provider("openai_batch")
    assert not isinstance(provider, NullBatchProvider)


# ---------------------------------------------------------------------------
# Stub providers raise with a clear next-step pointer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_stub_submit_raises_with_spec_pointer():
    """Anyone who reaches the stub gets a clear NEXT STEP, not a
    generic AttributeError or silent no-op."""
    provider = AnthropicBatchProvider()
    with pytest.raises(NotImplementedError, match="p0-spec.md §P0.1"):
        await provider.submit([])


@pytest.mark.asyncio
async def test_openai_stub_poll_raises_with_clear_message():
    provider = OpenAIBatchProvider()
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        await provider.poll(
            BatchSubmission(
                provider_batch_id="b1",
                provider="openai",
                submitted_at=datetime.now(timezone.utc),
            )
        )
