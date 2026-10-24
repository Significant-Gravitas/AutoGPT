"""Spraay provider configuration for AutoGPT blocks.

The Spraay gateway has no API keys: paid endpoints are settled per call with
x402 USDC micropayments. The credential stored for this provider is therefore
the private key of the EVM wallet that pays for those calls. Use a dedicated
hot wallet holding only a small USDC balance.
"""

from backend.sdk import APIKeyCredentials, BlockCostType, ProviderBuilder, SecretStr

SPRAAY_GATEWAY_BASE_URL = "https://gateway.spraay.app"
"""Base URL for the Spraay x402 payment gateway (all routes under /api/v1)."""

spraay_provider = (
    ProviderBuilder("spraay")
    .with_api_key(
        "SPRAAY_X402_PRIVATE_KEY",
        "Spraay x402 Payer Wallet Private Key",
    )
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
"""Spraay provider; the stored secret is the x402 payer wallet private key."""

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="spraay",
    title="Mock Spraay x402 payer key",
    api_key=SecretStr("0x" + "11" * 32),
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "id": TEST_CREDENTIALS.id,
    "provider": TEST_CREDENTIALS.provider,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}
