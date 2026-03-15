"""
Shared configuration for all Exa blocks using the new SDK pattern.
"""

from backend.sdk import BlockCostType, ProviderBuilder

from ._webhook import ExaWebhookManager

# Configure the Exa provider once for all blocks
exa = (
    ProviderBuilder("exa")
    .with_api_key("EXA_API_KEY", "Exa API Key")
    .with_webhook_manager(ExaWebhookManager)
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
