# Exa Websets Webhooks
<!-- MANUAL: file_description -->
Blocks for creating, reading, updating and deleting Exa Websets webhooks that send events to any URL.
<!-- END MANUAL -->

## Exa Create Webset Webhook

### What it is
Create an Exa Websets webhook that sends the chosen events to a URL, and get its one-time signing secret

### How it works
<!-- MANUAL: how_it_works -->
This block creates a webhook through the Websets API's `/v0/webhooks` for the URL and events you choose (1 to 19 events), with optional metadata. Exa returns the signing secret only in this response, so store it to verify deliveries via the Exa-Signature header.

Choosing the legacy `import.processing` event makes the block fail before calling Exa; use `import.completed`. To start a workflow from Websets events, use the Exa Webset Webhook trigger instead, which registers its own webhook. It records no Exa cost, so it is not charged. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | HTTPS URL Exa should deliver events to | str | Yes |
| events | Events to deliver (1-19). import.processing is legacy and is rejected; use import.completed. | List["webset.created" \| "webset.deleted" \| "webset.paused" \| "webset.idle" \| "webset.search.created" \| "webset.search.canceled" \| "webset.search.completed" \| "webset.search.updated" \| "import.created" \| "import.completed" \| "import.processing" \| "webset.item.created" \| "webset.item.enriched" \| "webset.export.created" \| "webset.export.completed" \| "monitor.created" \| "monitor.updated" \| "monitor.deleted" \| "monitor.run.created" \| "monitor.run.completed"] | Yes |
| metadata | Key-value pairs to store with the webhook | Dict[str, str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webhook_id | The new webhook's ID | str |
| secret | Signing secret for verifying deliveries (Exa-Signature header). Exa returns it only now, so store it safely. | str |
| webhook | The full webhook | Webhook |

### Possible use case
<!-- MANUAL: use_case -->
**External Integrations**: Send new webset items to your own service as they are found.

**Completion Notifications**: Notify a chat or ticketing system when a webset goes idle.

**Pipeline Hand-Off**: Tell a data pipeline when a Websets export has completed.
<!-- END MANUAL -->

---

## Exa Delete Webset Webhook

### What it is
Delete an Exa Websets webhook so Exa stops delivering to it

### How it works
<!-- MANUAL: how_it_works -->
This block deletes a webhook through the Websets API's `/v0/webhooks/{id}`, and Exa stops delivering to it. This cannot be undone.

It returns the webhook's ID and its status, which is inactive after deletion. It records no Exa cost, so it is not charged. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webhook_id | The webhook to delete. This cannot be undone. | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webhook_id | ID of the deleted webhook | str |
| status | The webhook's status, now inactive | str |

### Possible use case
<!-- MANUAL: use_case -->
**Decommissioning**: Remove a webhook when the endpoint it points to is retired.

**Duplicate Cleanup**: Delete extra webhooks found with the list block.

**Temporary Subscriptions**: Delete a webhook created for a one-off job once the job is done.
<!-- END MANUAL -->

---

## Exa Get Webset Webhook

### What it is
Get an Exa Websets webhook's URL, events, status and metadata

### How it works
<!-- MANUAL: how_it_works -->
This block fetches one webhook from the Websets API's `/v0/webhooks/{id}` and returns its status, subscribed events, URL and the full webhook including metadata. The signing secret is never returned again after creation.

It makes one request and does not wait or poll. It records no Exa cost, so it is not charged. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webhook_id | The webhook to fetch | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webhook_id | The webhook's ID | str |
| status | active or inactive | str |
| events | Subscribed events | List[str] |
| url | Where events are delivered | str |
| webhook | The full webhook | Webhook |

### Possible use case
<!-- MANUAL: use_case -->
**Configuration Check**: Confirm a webhook still points to the right URL.

**Subscription Review**: Check which events a webhook is subscribed to before changing them.

**Status Check**: See whether a webhook is still active.
<!-- END MANUAL -->

---

## Exa Update Webset Webhook

### What it is
Update an Exa Websets webhook's URL, events or metadata

### How it works
<!-- MANUAL: how_it_works -->
This block sends a PATCH to the Websets API's `/v0/webhooks/{id}` with only the fields you set; empty fields stay as they are. New events and metadata replace the existing ones rather than adding to them.

If no URL, events or metadata is set, the block fails before calling Exa. Choosing the legacy `import.processing` event also fails; use `import.completed`. It records no Exa cost, so it is not charged. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webhook_id | The webhook to update | str | Yes |
| url | New delivery URL; leave empty to keep it | str | No |
| events | Replace the subscribed events; leave empty to keep them. Events to deliver (1-19). import.processing is legacy and is rejected; use import.completed. | List["webset.created" \| "webset.deleted" \| "webset.paused" \| "webset.idle" \| "webset.search.created" \| "webset.search.canceled" \| "webset.search.completed" \| "webset.search.updated" \| "import.created" \| "import.completed" \| "import.processing" \| "webset.item.created" \| "webset.item.enriched" \| "webset.export.created" \| "webset.export.completed" \| "monitor.created" \| "monitor.updated" \| "monitor.deleted" \| "monitor.run.created" \| "monitor.run.completed"] | No |
| metadata | Replace the metadata; leave empty to keep it | Dict[str, str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webhook_id | The webhook's ID | str |
| webhook | The updated webhook | Webhook |

### Possible use case
<!-- MANUAL: use_case -->
**Endpoint Migration**: Point an existing webhook at a new URL instead of creating a new one.

**Event Changes**: Add or remove event types as your integration grows.

**Metadata Updates**: Change a webhook's owner or environment tags.
<!-- END MANUAL -->

---
