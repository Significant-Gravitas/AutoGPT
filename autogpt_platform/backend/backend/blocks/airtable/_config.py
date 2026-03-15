"""
Shared configuration for all Airtable blocks using the SDK pattern.
"""

from backend.sdk import BlockCostType, ProviderBuilder

from ._oauth import AirtableOAuthHandler, AirtableScope
from ._webhook import AirtableWebhookManager

# Configure the Airtable provider with API key authentication
airtable = (
    ProviderBuilder("airtable")
    .with_api_key("AIRTABLE_API_KEY", "Airtable Personal Access Token")
    .with_webhook_manager(AirtableWebhookManager)
    .with_base_cost(1, BlockCostType.RUN)
    .with_oauth(
        AirtableOAuthHandler,
        scopes=[
            v.value
            for v in [
                AirtableScope.DATA_RECORDS_READ,
                AirtableScope.DATA_RECORDS_WRITE,
                AirtableScope.SCHEMA_BASES_READ,
                AirtableScope.SCHEMA_BASES_WRITE,
                AirtableScope.WEBHOOK_MANAGE,
            ]
        ],
        client_id_env_var="AIRTABLE_CLIENT_ID",
        client_secret_env_var="AIRTABLE_CLIENT_SECRET",
    )
    .build()
)
