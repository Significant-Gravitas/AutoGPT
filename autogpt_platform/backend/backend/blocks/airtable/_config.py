"""
Shared configuration for all Airtable blocks using the SDK pattern.
"""

from backend.sdk import BlockCostType, ProviderBuilder

from ._webhook import AirtableWebhookManager

# Configure the Airtable provider with API key authentication
airtable = (
    ProviderBuilder("airtable")
    .with_api_key("AIRTABLE_API_KEY", "Airtable Personal Access Token")
    .with_webhook_manager(AirtableWebhookManager)
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
