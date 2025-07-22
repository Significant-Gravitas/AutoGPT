from backend.sdk import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    BlockType,
    BlockWebhookConfig,
    ProviderName,
    SchemaField,
)

from ._api import WebhookPayload
from ._config import airtable
from ._webhook import AirtableWebhookEvent


class AirtableWebhookTriggerBlock(Block):
    """
    Starts a flow whenever Airtable emits a webhook event.

    Thin wrapper just forwards the payloads one at a time to the next block.
    """

    class Input(BlockSchema):
        base_id: str = SchemaField(
            description="Airtable base ID"
        )
        table_id_or_name: str = SchemaField(
            description="Airtable table ID or name"
        )
        payloads: list[WebhookPayload] = SchemaField(
            description="Airtable webhook payload"
        )
        events: AirtableWebhookEvent = SchemaField(
            description="Airtable webhook event filter"
        )

    class Output(BlockSchema):
        payload: WebhookPayload = SchemaField(description="Airtable webhook payload")

    def __init__(self):
        example_payload = [
                    WebhookPayload(
                        actionMetadata={
                            "source": "client",
                            "sourceMetadata": {
                                "user": {
                                    "id": "usr00000000000000",
                                    "email": "foo@bar.com", 
                                    "permissionLevel": "create"
                                }
                            }
                        },
                        baseTransactionNumber=4,
                        payloadFormat="v0",
                        timestamp="2022-02-01T21:25:05.663Z"
                    )
                ]
        
        super().__init__(
            id="d0180ce6-ccb9-48c7-8256-b39e93e62801",
            description="Starts a flow whenever Airtable emits a webhook event",
            categories={BlockCategory.INPUT},
            input_schema=self.Input,
            output_schema=self.Output,
            block_type=BlockType.WEBHOOK,
            webhook_config=BlockWebhookConfig(
                provider=ProviderName("airtable"),
                webhook_type="not-used",
                event_filter_input="events",
                event_format="{event}",
                resource_format="{base_id}/{table_id_or_name}",
            ),
            test_input={
                "base_id": "app1234567890",
                "table_id_or_name": "table1234567890",
                "events": [AirtableWebhookEvent.TABLE_DATA, AirtableWebhookEvent.TABLE_FIELDS, AirtableWebhookEvent.TABLE_METADATA],
                "payloads": example_payload,
            },
            test_credentials=airtable.get_test_credentials(),
            test_output=[
                ("payload", example_payload[0]),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        for payload in input_data.payloads:
            yield "payload", payload