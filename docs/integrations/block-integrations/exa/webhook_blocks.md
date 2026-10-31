# Exa Webhook Blocks
<!-- MANUAL: file_description -->
Blocks for receiving webhook notifications from Exa webset events.
<!-- END MANUAL -->

## Exa Webset Webhook

### What it is
Triggers when Exa reports a Websets event, such as a new item, a finished search, a completed export or a monitor run

### How it works
<!-- MANUAL: how_it_works -->
This trigger is now enabled. When it is set up, the platform registers a webhook with Exa's Websets API for the events selected in Event Filter, and every delivery's `Exa-Signature` header is checked before the block runs. Registering needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account. Besides the webset, search, item, export and import events, it can now fire on the Websets monitor events: monitor created, updated and deleted, and monitor run created and completed. 'Import processing' is legacy: Exa no longer sends it, so it is never registered, and a filter with only that event selected is rejected.

Exa webhooks cover your whole Exa team, so when Webset ID is set, events about other websets are skipped; events that name no webset, such as imports, still pass through. Each run outputs the event type and ID, the affected webset, the full resource the event is about, its timestamp and its metadata.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | Only fire for events about this webset ID. Events that name no webset (imports, monitor runs) still fire. Leave empty to receive events for every webset on your Exa team. | str | No |
| event_filter | Configure which events to receive | WebsetEventFilter | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| event_type | Type of event that occurred, e.g. webset.item.created | str |
| event_id | Unique identifier for this event | str |
| webset_id | ID of the affected webset, or the Webset ID filter when the event names none | str |
| data | The full resource the event is about (webset, item, search, import or monitor) | Dict[str, Any] |
| timestamp | When the event occurred (ISO 8601) | str |
| metadata | Metadata attached to the resource the event is about | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Real-Time Processing**: Trigger workflows automatically when new items are added to websets without polling.

**Alert Systems**: Send a message when a webset search completes or an export is ready to download.

**Monitor Follow-Up**: Process new results each time a Websets monitor run completes.
<!-- END MANUAL -->

---
