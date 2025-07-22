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
from ._webhook import AirtableWebhookType


class AirtableWebhookTriggerBlock(Block):
    """
    Starts a flow whenever Airtable pings your webhook URL.

    Thin wrapper just forwards the payloads one at a time to the next block.
    """

    class Input(BlockSchema):
        base_id: str = SchemaField(description="Airtable base ID")
        table_id_or_name: str = SchemaField(description="Airtable table ID or name")
        payloads: list[WebhookPayload] = SchemaField(
            description="Airtable webhook payload"
        )
        events: AirtableWebhookType = SchemaField(
            description="Airtable webhook event filter"
        )

    class Output(BlockSchema):
        payload: WebhookPayload = SchemaField(description="Airtable webhook payload")

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
                webhook_type="airtable",
                event_filter_input="events",
                event_format="{event}",
                resource_format="{base_id}/{table_id_or_name}",
            ),
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        for payload in input_data.payloads:
            yield "payload", payload
