from backend.sdk import BlockCostType, ProviderBuilder

# Serply bills one credit per uncached search (cached responses are free).
# Packs start at $2.00 per 1,000 credits, so run() reports the upper bound of
# $0.002 per call as provider_cost and merge_stats populates
# NodeExecutionStats.provider_cost before billing reconciliation. 1000
# platform credits per USD matches the convention used by the other search
# providers (see tavily/_config.py).
serply = (
    ProviderBuilder("serply")
    .with_description("Google search results as JSON")
    .with_api_key("SERPLY_API_KEY", "Serply API Key")
    .with_base_cost(1000, BlockCostType.COST_USD)
    .build()
)
