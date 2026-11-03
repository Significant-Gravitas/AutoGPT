from backend.sdk import BlockCostType, ProviderBuilder

# Keenable search and fetch work without an API key on a per-IP rate-limited
# public tier; a key lifts the limit. Runs on the platform key are billed to the
# operator, so they carry the same flat 1-credit per-run base cost as other
# simple per-request API providers.
keenable = (
    ProviderBuilder("keenable")
    .with_description("Web search with page excerpts and URL-to-markdown fetch")
    .with_api_key("KEENABLE_API_KEY", "Keenable API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
