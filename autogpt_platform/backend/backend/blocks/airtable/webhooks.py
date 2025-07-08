"""
Airtable webhook management blocks.
"""

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import airtable


class AirtableFetchWebhookPayloadsBlock(Block):
    """
    Fetches accumulated event payloads for a webhook.

    Use this to pull the full change details after receiving a webhook notification,
    or run on a schedule to poll for changes.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID")
        webhook_id: str = SchemaField(
            description="The webhook ID to fetch payloads for"
        )
        cursor: int = SchemaField(
            description="Cursor position (0 = all payloads)", default=0
        )

    class Output(BlockSchema):
        payloads: list[dict] = SchemaField(description="Array of webhook payloads")
        next_cursor: int = SchemaField(description="Next cursor for pagination")
        might_have_more: bool = SchemaField(
            description="Whether there might be more payloads"
        )

    def __init__(self):
        super().__init__(
            id="7172db38-e338-4561-836f-9fa282c99949",
            description="Fetch webhook payloads from Airtable",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Fetch payloads from Airtable
        params = {}
        if input_data.cursor > 0:
            params["cursor"] = input_data.cursor

        response = await Requests().get(
            f"https://api.airtable.com/v0/bases/{input_data.base_id}/webhooks/{input_data.webhook_id}/payloads",
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
        )

        data = response.json()

        yield "payloads", data.get("payloads", [])
        yield "next_cursor", data.get("cursor", input_data.cursor)
        yield "might_have_more", data.get("mightHaveMore", False)


class AirtableRefreshWebhookBlock(Block):
    """
    Refreshes a webhook to extend its expiration by another 7 days.

    Webhooks expire after 7 days of inactivity. Use this block in a daily
    cron job to keep long-lived webhooks active.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID")
        webhook_id: str = SchemaField(description="The webhook ID to refresh")

    class Output(BlockSchema):
        expiration_time: str = SchemaField(
            description="New expiration time (ISO format)"
        )
        webhook: dict = SchemaField(description="Full webhook object")

    def __init__(self):
        super().__init__(
            id="5e82d957-02b8-47eb-8974-7bdaf8caff78",
            description="Refresh a webhook to extend its expiration",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Refresh the webhook
        response = await Requests().post(
            f"https://api.airtable.com/v0/bases/{input_data.base_id}/webhooks/{input_data.webhook_id}/refresh",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        webhook_data = response.json()

        yield "expiration_time", webhook_data.get("expirationTime", "")
        yield "webhook", webhook_data


class AirtableCreateWebhookBlock(Block):
    """
    Creates a new webhook for monitoring changes in an Airtable base.

    The webhook will send notifications to the specified URL when changes occur.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID to monitor")
        notification_url: str = SchemaField(
            description="URL to receive webhook notifications"
        )
        specification: dict = SchemaField(
            description="Webhook specification (filters, options)",
            default={
                "filters": {"dataTypes": ["tableData", "tableFields", "tableMetadata"]}
            },
        )

    class Output(BlockSchema):
        webhook: dict = SchemaField(description="Created webhook object")
        webhook_id: str = SchemaField(description="ID of the created webhook")
        mac_secret: str = SchemaField(
            description="MAC secret for signature verification"
        )
        expiration_time: str = SchemaField(description="Webhook expiration time")

    def __init__(self):
        super().__init__(
            id="b9f1f4ec-f4d1-4fbd-ab0b-b219c0e4da9a",
            description="Create a new Airtable webhook",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Create the webhook
        response = await Requests().post(
            f"https://api.airtable.com/v0/bases/{input_data.base_id}/webhooks",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "notificationUrl": input_data.notification_url,
                "specification": input_data.specification,
            },
        )

        webhook_data = response.json()

        yield "webhook", webhook_data
        yield "webhook_id", webhook_data.get("id", "")
        yield "mac_secret", webhook_data.get("macSecretBase64", "")
        yield "expiration_time", webhook_data.get("expirationTime", "")


class AirtableDeleteWebhookBlock(Block):
    """
    Deletes a webhook from an Airtable base.

    This will stop all notifications from the webhook.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="The Airtable base ID")
        webhook_id: str = SchemaField(description="The webhook ID to delete")

    class Output(BlockSchema):
        deleted: bool = SchemaField(
            description="Whether the webhook was successfully deleted"
        )

    def __init__(self):
        super().__init__(
            id="e4ded448-1515-4fe2-b93e-3e4db527df83",
            description="Delete an Airtable webhook",
            categories={BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        api_key = credentials.api_key.get_secret_value()

        # Delete the webhook
        response = await Requests().delete(
            f"https://api.airtable.com/v0/bases/{input_data.base_id}/webhooks/{input_data.webhook_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        # Check if deletion was successful
        deleted = response.status in [200, 204]

        yield "deleted", deleted
