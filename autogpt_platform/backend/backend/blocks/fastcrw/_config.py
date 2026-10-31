from backend.sdk import BlockCostType, ProviderBuilder

# fastCRW is a Firecrawl-API-compatible web data engine (single Rust binary;
# self-host or managed cloud at https://fastcrw.com). Because the API and data
# shapes match Firecrawl, these blocks mirror the Firecrawl provider exactly and
# simply point the client at the fastCRW base URL. Billing mirrors Firecrawl's
# credit model (1 credit ~= $0.001); each block's run() estimates USD spend and
# merge_stats populates NodeExecutionStats.provider_cost.
fastcrw = (
    ProviderBuilder("fastcrw")
    .with_description("Web scraping and crawling (fastCRW, Firecrawl-compatible)")
    .with_api_key("CRW_API_KEY", "fastCRW API Key")
    .with_base_cost(1000, BlockCostType.COST_USD)
    .build()
)
