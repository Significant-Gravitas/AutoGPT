from backend.sdk import BlockCostType, ProviderBuilder

# Search1API bills in credits: pay-as-you-go is $1 per 1,000 credits, and a
# basic search, news or crawl request costs 1 credit (each page fetched via
# crawl_results adds 1 more). Each block's run() reports USD spend through
# NodeExecutionStats.provider_cost; 1000 platform credits per USD matches the
# convention used by the other search providers (see tavily/_config.py).
search1api = (
    ProviderBuilder("search1api")
    .with_description("Multi-engine web search, news search and page crawling")
    .with_api_key("SEARCH1API_API_KEY", "Search1API API Key")
    .with_base_cost(1000, BlockCostType.COST_USD)
    .build()
)
