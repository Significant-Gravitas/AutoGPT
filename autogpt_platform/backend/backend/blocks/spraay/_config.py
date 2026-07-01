"""Spraay provider configuration for AutoGPT blocks.

Defines the Spraay provider using AutoGPT's ProviderBuilder pattern.
The provider uses API key authentication against the Spraay x402 gateway
at gateway.spraay.app.
"""

from backend.sdk import BlockCostType, ProviderBuilder

SPRAAY_GATEWAY_BASE_URL = "https://gateway.spraay.app"
"""Base URL for the Spraay x402 payment gateway."""

# Spraay uses x402 (HTTP 402 Payment Required) protocol.
# The API key authenticates the caller; payment is handled via USDC micropayments.
spraay_provider = (
    ProviderBuilder("spraay")
    .with_api_key("SPRAAY_API_KEY", "Spraay x402 Gateway API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
"""Configured Spraay provider with API key auth and per-run cost."""
