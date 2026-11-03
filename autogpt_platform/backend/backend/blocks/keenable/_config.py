from backend.sdk import ProviderBuilder

# Keenable search and fetch work without an API key on a per-IP rate-limited
# public tier; a key lifts the limit. No public per-call price is published
# for the keyed tier, so this provider registers no base_cost and its blocks
# report no provider_cost - same convention as other providers without a
# public rate.
keenable = (
    ProviderBuilder("keenable")
    .with_description("Web search with page excerpts and URL-to-markdown fetch")
    .with_api_key("KEENABLE_API_KEY", "Keenable API Key")
    .build()
)
