from backend.sdk import BlockCostType, ProviderBuilder

SPRAAY_GATEWAY_BASE_URL = "https://gateway.spraay.app"

# Spraay uses x402 (HTTP 402 Payment Required) protocol.
# The API key authenticates the caller; payment is handled via USDC micropayments.
spraay_provider = (
    ProviderBuilder("spraay")
    .with_api_key("SPRAAY_API_KEY", "Spraay x402 Gateway API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
