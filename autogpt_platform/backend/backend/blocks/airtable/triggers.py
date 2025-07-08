"""
Airtable webhook trigger blocks.
"""

from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockType,
    BlockWebhookConfig,
    CredentialsMetaInput,
    ProviderName,
    SchemaField,
)

from ._config import airtable


class AirtableWebhookTriggerBlock(Block):
    """
    Starts a flow whenever Airtable pings your webhook URL.

    If auto-fetch is enabled, it automatically fetches the full payloads
    after receiving the notification.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        webhook_url: str = SchemaField(
            description="URL to receive webhooks (auto-generated)",
            default="",
            hidden=True,
        )
        base_id: str = SchemaField(
            description="The Airtable base ID to monitor",
            default="",
        )
        table_id_or_name: str = SchemaField(
            description="Table ID or name to monitor (leave empty for all tables)",
            default="",
        )
        event_types: list[str] = SchemaField(
            description="Event types to listen for",
            default=["tableData", "tableFields", "tableMetadata"],
        )
        auto_fetch: bool = SchemaField(
            description="Automatically fetch full payloads after notification",
            default=True,
        )
        payload: dict = SchemaField(
            description="Webhook payload data",
            default={},
            hidden=True,
        )

    class Output(BlockSchema):
        ping: dict = SchemaField(description="Raw webhook notification body")
        headers: dict = SchemaField(description="Webhook request headers")
        verified: bool = SchemaField(
            description="Whether the webhook signature was verified"
        )
        # Fields populated when auto_fetch is True
        payloads: list[dict] = SchemaField(
            description="Array of change payloads (when auto-fetch is enabled)",
            default=[],
        )
        next_cursor: int = SchemaField(
            description="Next cursor for pagination (when auto-fetch is enabled)",
            default=0,
        )
        might_have_more: bool = SchemaField(
            description="Whether there might be more payloads (when auto-fetch is enabled)",
            default=False,
        )

    def __init__(self):
        super().__init__(
            id="d0180ce6-ccb9-48c7-8256-b39e93e62801",
            description="Starts a flow whenever Airtable pings your webhook URL",
            categories={BlockCategory.INPUT},
            input_schema=self.Input,
            output_schema=self.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName("airtable"),
                webhook_type="table_change",
                # event_filter_input="event_types",
                resource_format="{base_id}/{table_id_or_name}",
            ),
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        payload = input_data.payload

        # Extract headers from the webhook request (passed through kwargs)
        headers = kwargs.get("webhook_headers", {})

        # Check if signature was verified (handled by webhook manager)
        verified = True  # Webhook manager raises error if verification fails

        # Output basic webhook data
        yield "ping", payload
        yield "headers", headers
        yield "verified", verified

        # If auto-fetch is enabled and we have a cursor, fetch the full payloads
        if input_data.auto_fetch and payload.get("base", {}).get("id"):
            base_id = payload["base"]["id"]
            webhook_id = payload.get("webhook", {}).get("id", "")
            cursor = payload.get("cursor", 1)

            if webhook_id and cursor:
                # Get credentials from kwargs
                credentials = kwargs.get("credentials")
                if credentials:
                    # Fetch payloads using the Airtable API
                    api_key = credentials.api_key.get_secret_value()

                    from backend.sdk import Requests

                    response = await Requests().get(
                        f"https://api.airtable.com/v0/bases/{base_id}/webhooks/{webhook_id}/payloads",
                        headers={"Authorization": f"Bearer {api_key}"},
                        params={"cursor": cursor},
                    )

                    if response.status == 200:
                        data = response.json()
                        yield "payloads", data.get("payloads", [])
                        yield "next_cursor", data.get("cursor", cursor)
                        yield "might_have_more", data.get("mightHaveMore", False)
                    else:
                        # On error, still output empty payloads
                        yield "payloads", []
                        yield "next_cursor", cursor
                        yield "might_have_more", False
                else:
                    # No credentials, can't fetch
                    yield "payloads", []
                    yield "next_cursor", cursor
                    yield "might_have_more", False
        else:
            # Auto-fetch disabled or missing data
            yield "payloads", []
            yield "next_cursor", 0
            yield "might_have_more", False
