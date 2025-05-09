# """
# Module for Airtable webhook triggers.

# This module provides trigger blocks that respond to Airtable webhook events.
# """

# import logging
# from typing import Dict

# from strenum import StrEnum

# from backend.data.block import (
#     Block,
#     BlockCategory,
#     BlockManualWebhookConfig,
#     BlockOutput,
#     BlockSchema,
# )
# from backend.data.model import SchemaField
# from backend.integrations.providers import ProviderName

# logger = logging.getLogger(__name__)


# class AirtableWebhookEventType(StrEnum):
#     """Types of webhook events supported by Airtable."""

#     RECORDS_CREATED = "records:created"
#     RECORDS_UPDATED = "records:updated"
#     RECORDS_DELETED = "records:deleted"


# class AirtableWebhookTriggerBlock(Block):
#     """
#     A trigger block that responds to Airtable webhook events.
#     This block is activated when a webhook event is received from Airtable.
#     """

#     class Input(BlockSchema):
#         # The payload field is hidden because it's automatically populated by the webhook system
#         payload: Dict = SchemaField(hidden=True)

#     class Output(BlockSchema):
#         event_data: Dict = SchemaField(
#             description="The contents of the Airtable webhook event."
#         )
#         base_id: str = SchemaField(description="The ID of the Airtable base.")
#         table_id: str = SchemaField(description="The ID of the Airtable table.")
#         event_type: str = SchemaField(description="The type of event that occurred.")

#     def __init__(self):
#         super().__init__(
#             id="8c3b52d1-f7e9-4c5d-a6f1-60e937d94d2a",
#             description="This block will output the contents of an Airtable webhook event.",
#             categories={BlockCategory.DATA},
#             input_schema=AirtableWebhookTriggerBlock.Input,
#             output_schema=AirtableWebhookTriggerBlock.Output,
#             webhook_config=BlockManualWebhookConfig(
#                 provider=ProviderName.AIRTABLE,
#                 webhook_type=AirtableWebhookEventType.RECORDS_UPDATED,
#             ),
#             test_input=[
#                 {
#                     "payload": {
#                         "baseId": "app123",
#                         "tableId": "tbl456",
#                         "event": "records:updated",
#                         "data": {},
#                     }
#                 }
#             ],
#             test_output=[
#                 (
#                     "event_data",
#                     {
#                         "baseId": "app123",
#                         "tableId": "tbl456",
#                         "event": "records:updated",
#                         "data": {},
#                     },
#                 )
#             ],
#         )

#     def run(self, input_data: Input, **kwargs) -> BlockOutput:
#         """Process the Airtable webhook event and yield its contents."""
#         logger.info("Airtable webhook trigger received payload: %s", input_data.payload)
#         yield "event_data", input_data.payload
