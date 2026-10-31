# Exa Search Monitors Manage
<!-- MANUAL: file_description -->
Blocks for getting, listing and deleting Exa Search Monitors.
<!-- END MANUAL -->

## Exa Delete Search Monitor

### What it is
Permanently delete an Exa Search Monitor, stopping its runs

### How it works
<!-- MANUAL: how_it_works -->
This block sends `DELETE /monitors/{monitor_id}` to Exa, which permanently removes the monitor and stops its scheduled runs. It outputs the monitor as it was just before deletion; `monitor_id` falls back to the ID you passed if Exa returns none. The block does not charge for the call. The deletion cannot be undone, so the block is flagged as an irreversible action. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| monitor_id | The monitor to delete. This cannot be undone. | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| monitor_id | ID of the deleted monitor | str |
| monitor | The monitor as it was when deleted | Monitor |

### Possible use case
<!-- MANUAL: use_case -->
**Project Cleanup**: Delete a monitor once the topic it watched is no longer relevant.

**Workflow Teardown**: Remove the monitor a workflow created when that workflow is retired, so Exa stops posting to its webhook.

**Replacing a Monitor**: Delete an old monitor after creating a replacement with a better query.
<!-- END MANUAL -->

---

## Exa Get Search Monitor

### What it is
Get an Exa Search Monitor's query, schedule, webhook and status

### How it works
<!-- MANUAL: how_it_works -->
This block sends `GET /monitors/{monitor_id}` to Exa and returns the monitor's ID, status, next scheduled run and the full monitor (query, result count, domains, content options, period, output schema, metadata and webhook). `status` is `active`, `paused` or `disabled`; Exa sets `disabled` itself after repeated authentication failures. `next_run_at` is empty for a manual-only monitor. If Exa's response reports a `costDollars` total, the block bills it at 100 credits per USD; otherwise the call is free. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| monitor_id | The monitor to fetch | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| monitor_id | The monitor's ID | str |
| status | active, paused or disabled | str |
| next_run_at | Next scheduled run (ISO 8601); empty if manual-only | str |
| monitor | The full monitor | Monitor |

### Possible use case
<!-- MANUAL: use_case -->
**Health Check**: Check whether a monitor is still active or has been disabled by Exa before relying on its deliveries.

**Schedule Lookup**: Read `next_run_at` to tell users when the next batch of results will arrive.

**Config Audit**: Fetch a monitor's query, domains and webhook to confirm they match what a workflow expects.
<!-- END MANUAL -->

---

## Exa List Search Monitors

### What it is
List Exa Search Monitors, filtered by status, name or metadata

### How it works
<!-- MANUAL: how_it_works -->
This block sends `GET /monitors` to Exa with the status, name, limit and cursor filters you set; each metadata pair becomes a `metadata[key]=value` query parameter, and a monitor must match all of them. It returns the page as a list and also outputs each monitor one at a time. When `has_more` is true, pass `next_cursor` back as `cursor` to fetch the next page. The block does not charge for the call. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| status | Only list monitors in this status | "active" \| "paused" \| "disabled" | No |
| name | Only list monitors whose name contains this text (case-insensitive) | str | No |
| metadata | Only list monitors whose metadata has all of these key-value pairs (exact match) | Dict[str, str] | No |
| limit | Monitors per page (1-100) | int | No |
| cursor | Cursor from a previous page's next_cursor | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| monitors | Monitors on this page | List[SearchMonitorModel] |
| monitor | Each monitor, one at a time | Monitor |
| has_more | Whether another page exists | bool |
| next_cursor | Pass as cursor to fetch the next page | str |

### Possible use case
<!-- MANUAL: use_case -->
**Inventory**: List every monitor tagged with a given metadata value, such as a team or client ID.

**Status Review**: List disabled monitors to find ones that need their webhook or settings fixed.

**Bulk Processing**: Page through all monitors and run a follow-up block, such as Get Search Monitor Run, for each one.
<!-- END MANUAL -->

---
