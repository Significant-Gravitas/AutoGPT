from backend.sdk import (
    BaseModel,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    BlockType,
    BlockWebhookConfig,
    CredentialsMetaInput,
    ProviderName,
    SchemaField,
)

from ._api import WebhookPayload
from ._config import airtable


class AirtableEventSelector(BaseModel):
    """
    Selects the Airtable webhook event to trigger on.
    """

    tableData: bool = True
    tableFields: bool = True
    tableMetadata: bool = True


class AirtableWebhookTriggerBlock(Block):
    """
    Starts a flow whenever Airtable emits a webhook event.

    Thin wrapper just forwards the payloads one at a time to the next block.
    """

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = airtable.credentials_field(
            description="Airtable API credentials"
        )
        base_id: str = SchemaField(description="Airtable base ID")
        table_id_or_name: str = SchemaField(description="Airtable table ID or name")
        payload: dict = SchemaField(hidden=True, default_factory=dict)
        events: AirtableEventSelector = SchemaField(
            description="Airtable webhook event filter"
        )

    class Output(BlockSchemaOutput):
        payload: WebhookPayload = SchemaField(description="Airtable webhook payload")

    def __init__(self):
        example_payload = {
            "payloads": [
                {
                    "timestamp": "2022-02-01T21:25:05.663Z",
                    "baseTransactionNumber": 4,
                    "actionMetadata": {
                        "source": "client",
                        "sourceMetadata": {
                            "user": {
                                "id": "usr00000000000000",
                                "email": "foo@bar.com",
                                "permissionLevel": "create",
                            }
                        },
                    },
                    "payloadFormat": "v0",
                }
            ],
            "cursor": 5,
            "mightHaveMore": False,
        }

        super().__init__(
            # NOTE: This is disabled whilst the webhook system is finalised.
            disabled=False,
            id="d0180ce6-ccb9-48c7-8256-b39e93e62801",
            description="Starts a flow whenever Airtable emits a webhook event",
            categories={BlockCategory.INPUT, BlockCategory.DATA},
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
                "credentials": airtable.get_test_credentials().model_dump(),
                "base_id": "app1234567890",
                "table_id_or_name": "table1234567890",
                "events": AirtableEventSelector(
                    tableData=True,
                    tableFields=True,
                    tableMetadata=False,
                ).model_dump(),
                "payload": example_payload,
            },
            test_credentials=airtable.get_test_credentials(),
            test_output=[
                (
                    "payload",
                    WebhookPayload.model_validate(example_payload["payloads"][0]),
                ),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if len(input_data.payload["payloads"]) > 0:
            for item in input_data.payload["payloads"]:
                yield "payload", WebhookPayload.model_validate(item)
        else:
            yield "error", "No valid payloads found in webhook payload"
