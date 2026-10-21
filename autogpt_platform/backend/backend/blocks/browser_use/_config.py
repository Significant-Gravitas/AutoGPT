from backend.sdk import BlockCostType, ProviderBuilder

# Browser Use Cloud reports real spend: a run's totalCostUsd (LLM and web
# search) plus its session's browserCostUsd and proxyCostUsd. The block sums
# them into merge_stats so the COST_USD resolver bills actual usage; 1000
# platform credits per USD matches the other providers (see tavily/_config.py).
browser_use = (
    ProviderBuilder("browser_use")
    .with_description("Hosted browser agents and stealth cloud browsers")
    .with_api_key("BROWSER_USE_API_KEY", "Browser Use API Key")
    .with_base_cost(1000, BlockCostType.COST_USD)
    .build()
)
