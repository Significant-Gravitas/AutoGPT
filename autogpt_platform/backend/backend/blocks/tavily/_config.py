from backend.sdk import BlockCostType, ProviderBuilder

# Tavily bills in API credits (1 credit = $0.008 pay-as-you-go). Each block's
# run() estimates USD spend from the documented credit schedule (basic search
# 1 credit, advanced 2; extract 1 credit per 5 URLs; map 1 credit per 10
# pages) and merge_stats populates NodeExecutionStats.provider_cost before
# billing reconciliation. 1000 platform credits per USD matches the
# convention used by the other search providers (see firecrawl/_config.py).
tavily = (
    ProviderBuilder("tavily")
    .with_description("AI-native web search, extract, crawl and map")
    .with_api_key("TAVILY_API_KEY", "Tavily API Key")
    .with_base_cost(1000, BlockCostType.COST_USD)
    .build()
)
