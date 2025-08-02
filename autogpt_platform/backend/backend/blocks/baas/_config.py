"""
Shared configuration for all Meeting BaaS blocks using the SDK pattern.
"""

from backend.sdk import BlockCostType, ProviderBuilder

from ._webhook import BaasWebhookManager

# Configure the Meeting BaaS provider with API key authentication
baas = (
    ProviderBuilder("baas")
    .with_api_key("MEETING_BAAS_API_KEY", "Meeting BaaS API Key")
    .with_webhook_manager(BaasWebhookManager)
    .with_base_cost(5, BlockCostType.RUN)  # Higher cost for meeting recording service
    .build()
)
