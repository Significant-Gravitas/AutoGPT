from backend.sdk import BlockCostType, ProviderBuilder

# Each Stagehand call spawns a Browserbase session ($1/h ≈ $0.00028/s) AND
# issues an LLM call on the chosen model (GPT-4.1 / Claude Sonnet tier).
# Block walltime is a reasonable proxy for both (session lifetime == block
# lifetime; LLM latency is subsumed). 2 cr/s (~$0.02/s) covers the session
# infra + a small LLM call with margin. Interim until we pipe Browserbase
# session-seconds and the underlying LLM's token counts into provider_cost.
stagehand = (
    ProviderBuilder("stagehand")
    .with_api_key("STAGEHAND_API_KEY", "Stagehand API Key")
    .with_base_cost(2, BlockCostType.SECOND)
    .build()
)
