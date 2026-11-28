from backend.sdk import BlockCostType, ProviderBuilder

# SearchApi bills per successful (HTTP 200) search; failed requests are free.
# The published plans range from $4 per 1,000 searches (Developer) down to
# $1 per 1,000 at volume, so run() reports the top of that range, $0.004 per
# call, as provider_cost and merge_stats populates
# NodeExecutionStats.provider_cost before billing reconciliation. 1000
# platform credits per USD matches the convention used by the other search
# providers (see tavily/_config.py).
searchapi = (
    ProviderBuilder("searchapi")
    .with_description("Google search results as structured JSON")
    .with_api_key("SEARCHAPI_API_KEY", "SearchApi API Key")
    .with_base_cost(1000, BlockCostType.COST_USD)
    .build()
)
