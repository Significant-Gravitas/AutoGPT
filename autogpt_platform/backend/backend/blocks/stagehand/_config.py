from backend.sdk import BlockCostType, ProviderBuilder

# 1 credit per 3 walltime seconds. Block walltime proxies for the
# Browserbase session lifetime + the LLM call it issues. Interim until
# the block emits real provider_cost (USD) via merge_stats and migrates
# to COST_USD.
stagehand = (
    ProviderBuilder("stagehand")
    .with_description("AI browser automation")
    .with_api_key("STAGEHAND_API_KEY", "Stagehand API Key")
    .with_base_cost(1, BlockCostType.SECOND, cost_divisor=3)
    .build()
)
