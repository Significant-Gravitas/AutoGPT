"""Pick a BatchProvider for a given ExecutionPath + config state.

Single seam the orchestrator calls when it routes to a batch path.
Today the AnthropicBatchProvider and OpenAIBatchProvider are stubs;
this factory always returns the Null provider so the orchestrator's
batch path short-circuits to sync_baseline cleanly when the stubs
are detected. As soon as the real adapters land, flip the
``USE_STUB_BATCH_PROVIDERS`` flag off and the orchestrator picks
them up automatically.

The factory deliberately does NOT read provider API keys here —
``resolve_dream_execution_path`` in routing.py already gates on
``has_anthropic_key`` / ``has_openai_key`` upstream. If we get to
this factory, the routing decision said "use this batch path"; our
job is only to hand back a working provider for that path.
"""

from __future__ import annotations

import logging

from ..routing import ExecutionPath
from .anthropic_provider import AnthropicBatchProvider
from .openai_provider import OpenAIBatchProvider
from .provider import BatchProvider, NullBatchProvider

logger = logging.getLogger(__name__)


# The orchestrator's Anthropic batch path now goes directly through
# ``backend/util/llm/providers.call_provider(execution_mode="batch")``;
# this factory is preserved only for callers that still want the
# Protocol-shaped provider abstraction (P-0.1 stub callers, future
# OpenAI batch path). Flipped to ``False`` as of Step 5 of the
# plan in ``plans/idempotent-launching-moth.md`` so the orchestrator
# isn't confused into thinking the providers are stubbed.
USE_STUB_BATCH_PROVIDERS = False


def get_batch_provider(execution_path: ExecutionPath) -> BatchProvider:
    """Return the BatchProvider for the chosen execution path.

    ``sync_baseline`` always returns the Null provider — sync runs
    don't use batch at all; the orchestrator gates on the path
    before calling this. The function still accepts sync_baseline to
    avoid a special case at the call site.
    """
    if execution_path == "sync_baseline":
        return NullBatchProvider()

    if USE_STUB_BATCH_PROVIDERS:
        logger.info(
            "dream batch: execution_path=%s but providers are stubbed; "
            "returning NullBatchProvider — orchestrator should fall back "
            "to sync_baseline.",
            execution_path,
        )
        return NullBatchProvider()

    if execution_path == "anthropic_batch":
        return AnthropicBatchProvider()
    if execution_path == "openai_batch":
        return OpenAIBatchProvider()
    return NullBatchProvider()
