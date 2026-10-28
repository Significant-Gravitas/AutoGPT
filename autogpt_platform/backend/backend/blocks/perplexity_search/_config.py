"""Provider configuration for the Perplexity Search API block."""

from backend.sdk import BlockCostType, ProviderBuilder

# Perplexity bills the Search API per successful request: $0.005 for a web
# search and $0.001 for a fast search (docs.perplexity.ai/getting-started/
# pricing). run() records the matching USD amount as provider_cost and
# 1000 platform credits per USD matches the other search providers
# (see tavily/_config.py and firecrawl/_config.py).
perplexity = (
    ProviderBuilder("perplexity")
    .with_description("Web search with ranked results and snippets")
    .with_api_key("PERPLEXITY_API_KEY", "Perplexity API Key")
    .with_base_cost(1000, BlockCostType.COST_USD)
    .build()
)
