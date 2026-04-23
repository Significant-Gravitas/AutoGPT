from backend.sdk import BlockCostType, ProviderBuilder

# Each Stagehand call spawns a Browserbase session (~$0.00028/s infra)
# AND issues an LLM call (~$0.001/call on GPT-4.1-mini tier). Median
# real cost ≈ $0.001/s. 1 credit per 3 walltime seconds (≈ $0.0033/s)
# covers real cost with ~3x margin. Block walltime is a reasonable
# proxy for both streams (session lifetime == block lifetime; LLM
# latency is subsumed). Interim until the block emits real provider_cost
# (USD) via merge_stats, at which point migrate to COST_USD.
stagehand = (
    ProviderBuilder("stagehand")
    .with_api_key("STAGEHAND_API_KEY", "Stagehand API Key")
    .with_base_cost(1, BlockCostType.SECOND, cost_divisor=3)
    .build()
)
