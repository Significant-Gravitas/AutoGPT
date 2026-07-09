"""Batch-API integration for dream passes.

Anthropic and OpenAI both offer a Message Batch API that's 50% cheaper
than the synchronous endpoint but takes hours instead of seconds. The
dream pass is exactly the workload that wants this — it runs at 3 AM,
nobody is waiting on it, and 50% off three Sonnet/Opus calls per user
per night is meaningful at production scale.

Module layout (per ``dream/p0-spec.md`` §P0.1):
  * models.py            — typed BatchRequest / BatchResult / BatchStatus
  * provider.py          — BatchProvider Protocol + NullBatchProvider
  * anthropic_provider.py — TODO: real Anthropic AsyncMessageBatch wrapper
  * openai_provider.py   — TODO: real OpenAI client.batches wrapper
  * factory.py           — pick a provider for a given ExecutionPath

Today the dream orchestrator only wires sync_baseline; the
provider-aware path is staged here so the orchestrator can adopt it
incrementally without another big refactor.
"""
