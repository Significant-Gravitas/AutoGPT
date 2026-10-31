"""Provider configuration for the Perplexity Search API block."""

from backend.sdk import BlockCostType, ProviderBuilder

# Perplexity bills each successful request ($0.005 web, $0.001 fast), and run()
# records that charge as provider_cost. The block runs only on the user's own
# Perplexity key, which the platform doesn't bill: "perplexity" isn't in
# RUNNABLE_SDK_DEFAULT_PROVIDERS, so a platform "perplexity-default" key is
# refused. The COST_USD base cost (150 credits per USD) would only apply if
# Perplexity were added there.
perplexity = (
    ProviderBuilder("perplexity")
    .with_description("Web search with ranked results and snippets")
    .with_api_key("PERPLEXITY_API_KEY", "Perplexity API Key")
    .with_base_cost(150, BlockCostType.COST_USD)
    .build()
)
