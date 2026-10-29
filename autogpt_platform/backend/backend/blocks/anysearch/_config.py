from backend.sdk import ProviderBuilder

# AnySearch does not publish per-call pricing (Free tier: 1,000 requests/day;
# the Professional plan is marked "coming soon"), so this provider registers no
# base_cost and its blocks report no provider_cost - same convention as other
# providers without a public rate. One-line addition once rates are published.
anysearch = (
    ProviderBuilder("anysearch")
    .with_description("AI-native web search, parallel search and URL extraction")
    .with_api_key("ANYSEARCH_API_KEY", "AnySearch API Key")
    .build()
)
