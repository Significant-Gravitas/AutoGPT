from backend.sdk import BlockCostType, ProviderBuilder

firecrawl = (
    ProviderBuilder("firecrawl")
    .with_api_key("FIRECRAWL_API_KEY", "Firecrawl API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
