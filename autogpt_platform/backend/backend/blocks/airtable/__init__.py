"""
Airtable integration for AutoGPT Platform.

This integration provides comprehensive access to the Airtable Web API,
including:
- Webhook triggers and management
- Record CRUD operations
- Schema and table management
"""

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

__all__ = [
    # Webhook Triggers
    "AirtableWebhookTriggerBlock",
    # Record Operations
    "AirtableCreateRecordsBlock",
    "AirtableDeleteRecordsBlock",
    "AirtableGetRecordBlock",
    "AirtableListRecordsBlock",
    "AirtableUpdateRecordsBlock",
    "AirtableUpsertRecordsBlock",
    # Schema & Table Management
    "AirtableAddFieldBlock",
    "AirtableCreateTableBlock",
    "AirtableDeleteFieldBlock",
    "AirtableDeleteTableBlock",
    "AirtableListSchemaBlock",
    "AirtableUpdateFieldBlock",
    "AirtableUpdateTableBlock",
]
