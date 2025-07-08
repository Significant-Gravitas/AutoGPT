"""
Airtable integration for AutoGPT Platform.

This integration provides comprehensive access to the Airtable Web API,
including:
- Webhook triggers and management
- Record CRUD operations
- Attachment uploads
- Schema and table management
- Metadata operations
"""

# Attachments
from .attachments import AirtableUploadAttachmentBlock

# Metadata
from .metadata import (
    AirtableGetViewBlock,
    AirtableListBasesBlock,
    AirtableListViewsBlock,
)

# Record Operations
from .records import (
    AirtableCreateRecordsBlock,
    AirtableDeleteRecordsBlock,
    AirtableGetRecordBlock,
    AirtableListRecordsBlock,
    AirtableUpdateRecordsBlock,
    AirtableUpsertRecordsBlock,
)

# Schema & Table Management
from .schema import (
    AirtableAddFieldBlock,
    AirtableCreateTableBlock,
    AirtableDeleteFieldBlock,
    AirtableDeleteTableBlock,
    AirtableListSchemaBlock,
    AirtableUpdateFieldBlock,
    AirtableUpdateTableBlock,
)

# Webhook Triggers
from .triggers import AirtableWebhookTriggerBlock

# Webhook Management
from .webhooks import (
    AirtableCreateWebhookBlock,
    AirtableDeleteWebhookBlock,
    AirtableFetchWebhookPayloadsBlock,
    AirtableRefreshWebhookBlock,
)

__all__ = [
    # Webhook Triggers
    "AirtableWebhookTriggerBlock",
    # Webhook Management
    "AirtableCreateWebhookBlock",
    "AirtableDeleteWebhookBlock",
    "AirtableFetchWebhookPayloadsBlock",
    "AirtableRefreshWebhookBlock",
    # Record Operations
    "AirtableCreateRecordsBlock",
    "AirtableDeleteRecordsBlock",
    "AirtableGetRecordBlock",
    "AirtableListRecordsBlock",
    "AirtableUpdateRecordsBlock",
    "AirtableUpsertRecordsBlock",
    # Attachments
    "AirtableUploadAttachmentBlock",
    # Schema & Table Management
    "AirtableAddFieldBlock",
    "AirtableCreateTableBlock",
    "AirtableDeleteFieldBlock",
    "AirtableDeleteTableBlock",
    "AirtableListSchemaBlock",
    "AirtableUpdateFieldBlock",
    "AirtableUpdateTableBlock",
    # Metadata
    "AirtableGetViewBlock",
    "AirtableListBasesBlock",
    "AirtableListViewsBlock",
]
