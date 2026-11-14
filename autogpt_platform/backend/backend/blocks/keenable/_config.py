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

# Anonymous runs drop any selected credential and call the keyless endpoint, so
# the platform-key cost only applies when the node is set to auth=api_key.
# (`register_provider_costs_for_block` adds the credential-id filter on top.)
keenable.base_costs = [
    cost.model_copy(update={"cost_filter": {**cost.cost_filter, "auth": "api_key"}})
    for cost in keenable.base_costs
]
