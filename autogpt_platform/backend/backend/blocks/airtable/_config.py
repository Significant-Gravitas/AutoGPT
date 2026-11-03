"""
Shared configuration for all Airtable blocks using the SDK pattern.
"""

from typing import Literal

from backend.sdk import BlockCostType, ProviderBuilder, SchemaField

from ._oauth import AirtableOAuthHandler, AirtableScope
from ._webhook import AirtableWebhookManager

# Configure the Airtable provider with API key authentication
airtable = (
    ProviderBuilder("airtable")
    .with_description("Bases, tables, and records")
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


def base_id_field(access: Literal["read", "edit", "create"]) -> str:
    """A base the user picks from the ones their connection can reach.

    Copilot cards render this as a list of the account's bases, so nobody is
    asked for an id from a URL; anywhere else it stays a plain text input.
    ``access`` is the least a base must allow for the block to work there:
    Airtable lets only a base's creators change its tables and fields or
    watch it with a webhook, editors write records, and anyone can read.
    """
    return SchemaField(
        title="Airtable base",
        description=(
            "The Airtable base to use: a set of tables, like a spreadsheet "
            "file with several sheets."
        ),
        format="airtable-base-picker",
        json_schema_extra={
            "resource_picker": {"provider": "airtable", "access": access}
        },
    )
