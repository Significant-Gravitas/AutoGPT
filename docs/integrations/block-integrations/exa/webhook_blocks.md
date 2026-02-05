# Exa Webhook Blocks
<!-- MANUAL: file_description -->
Blocks for receiving webhook notifications from Exa webset events.
<!-- END MANUAL -->

## Exa Webset Webhook

### What it is
Receive webhook notifications for Exa webset events

### How it works
<!-- MANUAL: how_it_works -->
This block acts as a webhook receiver for Exa webset events. When events occur on your websets (like new items found, searches completed, or enrichments finished), Exa sends notifications to this webhook endpoint.

The block can filter events by webset ID and event type. It parses incoming webhook payloads and outputs structured event data including the event type, affected webset, and event-specific details.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The webset ID to monitor (optional, monitors all if empty) | str | No |
| event_filter | Configure which events to receive | WebsetEventFilter | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| event_type | Type of event that occurred | str |
| event_id | Unique identifier for this event | str |
| webset_id | ID of the affected webset | str |
| data | Event-specific data | Dict[str, Any] |
| timestamp | When the event occurred | str |
| metadata | Additional event metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Real-Time Processing**: Trigger workflows automatically when new items are added to websets without polling.

**Alert Systems**: Receive instant notifications when webset searches find new relevant results.

**Integration Pipelines**: Build event-driven integrations that react to webset changes in real time.
<!-- END MANUAL -->

---
