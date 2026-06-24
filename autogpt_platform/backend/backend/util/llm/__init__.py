"""Shared LLM dispatch infrastructure.

Owns the per-provider SDK call helper (``providers.call_provider``)
so that multiple LLM call sites in the backend — the block-layer
``_llm_call``, the dream-pass ``structured_completion``, the copilot
chat dispatch, future server-internal callers — route through one
implementation. Adding a new provider, a new execution mode (batch,
flex), or new structured-output handling lands once and every caller
picks it up.

This module is intentionally agnostic to the caller's surrounding
framework: it doesn't depend on the block layer's ``LlmModel`` enum,
the executor's ``NodeExecutionStats``, or the dream-pass's Pydantic
validation. Callers wrap it with their own typed entry points.
"""
