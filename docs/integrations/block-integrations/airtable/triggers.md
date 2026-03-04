# Airtable Triggers
<!-- MANUAL: file_description -->
Blocks for triggering workflows based on Airtable events like record creation, updates, and deletions.
<!-- END MANUAL -->

## Airtable Webhook Trigger

### What it is
Starts a flow whenever Airtable emits a webhook event

### How it works
<!-- MANUAL: how_it_works -->
This block subscribes to Airtable webhook events for a specific base and table. When records are created, updated, or deleted, Airtable sends a webhook notification that triggers your workflow.

You specify which events to listen for using the event selector. The webhook payload includes details about the changed records and the type of change that occurred.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| base_id | Airtable base ID | str | Yes |
| table_id_or_name | Airtable table ID or name | str | Yes |
| events | Airtable webhook event filter | AirtableEventSelector | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| payload | Airtable webhook payload | WebhookPayload |

### Possible use case
<!-- MANUAL: use_case -->
**Real-Time Sync**: Automatically sync Airtable changes to other systems like CRMs or databases.

**Notification Workflows**: Send alerts when specific records are created or modified in Airtable.

**Automated Processing**: Trigger document generation or emails when new entries are added to a table.
<!-- END MANUAL -->

---
