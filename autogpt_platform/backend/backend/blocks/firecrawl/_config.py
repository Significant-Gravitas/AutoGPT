from backend.sdk import BlockCostType, ProviderBuilder

# Firecrawl bills in its own credits (1 credit ≈ $0.001). Each block's
# run() estimates USD spend from the operation (pages scraped, limit,
# credits_used on ExtractResponse) and merge_stats populates
# NodeExecutionStats.provider_cost before billing reconciliation. 1000
# platform credits per USD means 1 platform credit per Firecrawl credit
# — roughly matches our existing per-call tier for single-page scrape.
firecrawl = (
    ProviderBuilder("firecrawl")
    .with_description("Web scraping and crawling")
    .with_api_key("FIRECRAWL_API_KEY", "Firecrawl API Key")
    .with_base_cost(1000, BlockCostType.COST_USD)
    .build()
)
