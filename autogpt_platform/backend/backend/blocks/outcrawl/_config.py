from backend.sdk import BlockCostType, ProviderBuilder

# Outcrawl bills 1 credit per page, whether the page was scraped directly or
# rendered as a search hit; a plain search is 1 credit per request. Each
# block's run() reads the credits from the response's usage report and
# merge_stats populates NodeExecutionStats.provider_cost before billing
# reconciliation. 1000 platform credits per USD matches the convention used
# by the other search providers (see firecrawl/_config.py).
outcrawl = (
    ProviderBuilder("outcrawl")
    .with_description("Web scraping and search for AI agents")
    .with_api_key("OUTCRAWL_API_KEY", "Outcrawl API Key")
    .with_base_cost(1000, BlockCostType.COST_USD)
    .build()
)
