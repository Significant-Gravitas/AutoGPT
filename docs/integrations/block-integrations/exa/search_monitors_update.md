# Exa Search Monitors Update
<!-- MANUAL: file_description -->
Blocks for changing an Exa Search Monitor's status, search, schedule, output format or webhook.
<!-- END MANUAL -->

## Exa Update Search Monitor

### What it is
Update an Exa Search Monitor: pause or resume it, or change its query, schedule, output format or webhook

### How it works
<!-- MANUAL: how_it_works -->
This block sends `PATCH /monitors/{monitor_id}` to Exa with only the fields you set; everything left empty keeps its current value, and the block fails if nothing is set. `remove_schedule` sends a null trigger, making the monitor manual-only, and wins over `period`; `clear_metadata` removes all metadata and wins over `metadata`. A new `period` follows the same rule as on create (a whole number followed by `h` or `d`). Leaving a domain or webhook event list, the content options or the output schema empty keeps the current value, so they cannot be cleared here. If Exa's response reports a `costDollars` total, the block bills it at 100 credits per USD; otherwise the call is free. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| monitor_id | The monitor to update | str | Yes |
| status | active runs on schedule; paused runs only when triggered. Leave empty to keep the current status. | "active" \| "paused" | No |
| query | New search query | str | No |
| period | How often the monitor runs: a whole number followed by h (hours) or d (days), minimum 1h, e.g. 6h, 1d or 7d. The schedule is anchored to the creation time and a run may start up to 30 minutes late. Leave empty to keep the current schedule. | str | No |
| remove_schedule | Remove the schedule so the monitor only runs when triggered (overrides the run interval) | bool | No |
| name | New name | str | No |
| num_results | Search results per run (1-100) | int | No |
| include_domains | Replace the included domains. Leave empty to keep the current list | List[str] | No |
| exclude_domains | Replace the excluded domains. Leave empty to keep the current list | List[str] | No |
| contents | Replace the per-result content options. Leave empty to keep the current ones | ContentSettings | No |
| output_schema | Replace the output format: a JSON Schema, or {"type": "text"} for a written summary. Leave empty to keep the current one | Dict[str, Any] | No |
| metadata | Replace the metadata | Dict[str, str] | No |
| clear_metadata | Remove all metadata (overrides metadata) | bool | No |
| webhook_url | New webhook URL. HTTPS URL Exa posts each run to. To handle runs in a workflow, add an 'Exa Search Monitor Trigger' block to it, save, and paste that block's webhook URL here. Must be publicly reachable (no localhost or private IPs); redirects are not followed. | str | No |
| webhook_events | Replace the webhook's subscribed events. Leave empty to keep the current ones | List["monitor.created" \| "monitor.updated" \| "monitor.deleted" \| "monitor.run.created" \| "monitor.run.completed"] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| monitor_id | The monitor's ID | str |
| status | Status after the update | str |
| next_run_at | Next scheduled run (ISO 8601); empty if manual-only | str |
| monitor | The updated monitor | Monitor |

### Possible use case
<!-- MANUAL: use_case -->
**Pause and Resume**: Pause a monitor during a quiet period and set it back to active later.

**Query Refinement**: Tighten a monitor's query or domains after early runs return off-topic results.

**Webhook Move**: Point a monitor at a different Exa Search Monitor Trigger block's webhook URL when a workflow is rebuilt.
<!-- END MANUAL -->

---
