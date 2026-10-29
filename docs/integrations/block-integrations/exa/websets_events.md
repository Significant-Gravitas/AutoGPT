# Exa Websets Events
<!-- MANUAL: file_description -->
Blocks for reading the event history Exa records for your Websets and Websets monitors.
<!-- END MANUAL -->

## Exa Get Webset Event

### What it is
Get one Exa Websets event and the resource it is about

### How it works
<!-- MANUAL: how_it_works -->
This block fetches one event from the Websets API's `/v0/events/{id}` and returns its ID, type, time and the full resource it is about (a webset, item, search, import or monitor). Exa keeps events for 60 days.

It makes one request and does not wait or poll. It records no Exa cost, so it is not charged. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| event_id | The event to fetch | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| event_id | The event's ID | str |
| event_type | e.g. webset.item.created | str |
| created_at | When it happened (ISO 8601) | str |
| data | The full resource the event is about (webset, item, search, import or monitor) | Dict[str, Any] |
| event | The full event | Event |

### Possible use case
<!-- MANUAL: use_case -->
**Webhook Follow-Up**: Look up the full event behind an event ID received from a webhook delivery.

**Delivery Debugging**: Fetch the event named in a failed webhook attempt to see what was not delivered.

**Item Inspection**: Read the webset item attached to a webset.item.created event.
<!-- END MANUAL -->

---

## Exa List Webset Events

### What it is
List Exa Websets events (new items, finished searches, monitor runs and more) from the last 60 days

### How it works
<!-- MANUAL: how_it_works -->
This block lists events from the Websets API's `/v0/events`, newest first, from the last 60 days. Each selected type is sent as its own filter, and times without a timezone are treated as UTC. Each call returns one page of up to 200 events (25 by default); pass `next_cursor` back as `cursor` for the next page.

Selecting the legacy `import.processing` type makes the block fail with a message to use `import.completed`. It is a polling alternative to the Exa Webset Webhook trigger. It records no Exa cost, so it is not charged. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| types | Only list these event types; leave empty for all | List["webset.created" \| "webset.deleted" \| "webset.paused" \| "webset.idle" \| "webset.search.created" \| "webset.search.canceled" \| "webset.search.completed" \| "webset.search.updated" \| "import.created" \| "import.completed" \| "import.processing" \| "webset.item.created" \| "webset.item.enriched" \| "webset.export.created" \| "webset.export.completed" \| "monitor.created" \| "monitor.updated" \| "monitor.deleted" \| "monitor.run.created" \| "monitor.run.completed"] | No |
| created_after | Only events at or after this time (UTC if no timezone) | str (date-time) | No |
| created_before | Only events at or before this time (UTC if no timezone) | str (date-time) | No |
| limit | Events per page (1-200) | int | No |
| cursor | Cursor from a previous page's next_cursor | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| events | Events on this page | List[WebsetEventModel] |
| event | Each event, one at a time | Event |
| has_more | Whether another page exists | bool |
| next_cursor | Pass as cursor to fetch the next page | str |

### Possible use case
<!-- MANUAL: use_case -->
**Scheduled Polling**: Check every hour for new webset.item.created events and process the new items.

**Activity Log**: Collect all events from the past week into a sheet or report.

**Missed Event Recovery**: Replay events from a time window when a webhook endpoint was down.
<!-- END MANUAL -->

---
