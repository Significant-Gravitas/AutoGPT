# Exa Websets Webhook Attempts
<!-- MANUAL: file_description -->
Blocks for listing Exa Websets webhooks and their delivery attempts.
<!-- END MANUAL -->

## Exa List Webset Webhook Attempts

### What it is
List an Exa Websets webhook's delivery attempts, with status codes and responses, to debug failed deliveries

### How it works
<!-- MANUAL: how_it_works -->
This block lists delivery attempts for one webhook from the Websets API's `/v0/webhooks/{id}/attempts`, newest first. Each attempt has the event, URL, whether it succeeded, the response status code, headers and body, and the attempt number.

You can filter by one event type and by success or failure. Each call returns one page of up to 200 attempts (25 by default); pass `next_cursor` back as `cursor` for the next page. It records no Exa cost, so it is not charged. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webhook_id | The webhook whose deliveries to list | str | Yes |
| event_type | Only list attempts for this event type | "webset.created" \| "webset.deleted" \| "webset.paused" \| "webset.idle" \| "webset.search.created" \| "webset.search.canceled" \| "webset.search.completed" \| "webset.search.updated" \| "import.created" \| "import.completed" \| "import.processing" \| "webset.item.created" \| "webset.item.enriched" \| "webset.export.created" \| "webset.export.completed" \| "monitor.created" \| "monitor.updated" \| "monitor.deleted" \| "monitor.run.created" \| "monitor.run.completed" | No |
| successful | Only list successful (true) or failed (false) attempts; leave empty for both | bool | No |
| limit | Attempts per page (1-200) | int | No |
| cursor | Cursor from a previous page's next_cursor | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| attempts | Delivery attempts on this page, newest first | List[WebhookAttemptModel] |
| attempt | Each attempt, one at a time | Attempt |
| has_more | Whether another page exists | bool |
| next_cursor | Pass as cursor to fetch the next page | str |

### Possible use case
<!-- MANUAL: use_case -->
**Failed Delivery Debugging**: List failed attempts to see the status codes and error bodies your endpoint returned.

**Endpoint Health Check**: Count recent failed attempts and alert when a webhook keeps failing.

**Event-Specific Review**: Check whether webset.idle deliveries reached your endpoint after a long search.
<!-- END MANUAL -->

---

## Exa List Webset Webhooks

### What it is
List your team's active Exa Websets webhooks

### How it works
<!-- MANUAL: how_it_works -->
This block lists your team's active webhooks from the Websets API's `/v0/webhooks`, with each one's URL, subscribed events, status and metadata. Signing secrets are not included.

Each call returns one page of up to 200 webhooks (25 by default); pass `next_cursor` back as `cursor` for the next page. It records no Exa cost, so it is not charged. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| limit | Webhooks per page (1-200) | int | No |
| cursor | Cursor from a previous page's next_cursor | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webhooks | Webhooks on this page | List[WebsetWebhookModel] |
| webhook | Each webhook, one at a time | Webhook |
| has_more | Whether another page exists | bool |
| next_cursor | Pass as cursor to fetch the next page | str |

### Possible use case
<!-- MANUAL: use_case -->
**Webhook Inventory**: See which URLs receive Websets events and for which event types.

**Cleanup**: Find old or duplicate webhooks to pass to the delete block.

**Configuration Check**: Confirm a webhook exists before creating a new one for the same URL.
<!-- END MANUAL -->

---
