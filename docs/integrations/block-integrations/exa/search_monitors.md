# Exa Search Monitors
<!-- MANUAL: file_description -->
Blocks for creating Exa Search Monitors that re-run a search on a schedule and post new results to a webhook.
<!-- END MANUAL -->

## Exa Create Search Monitor

### What it is
Create an Exa Search Monitor that re-runs a search on a schedule and posts only new results to a webhook (pair it with the Exa Search Monitor Trigger block)

### How it works
<!-- MANUAL: how_it_works -->
This block sends a single `POST /monitors` request to Exa and returns as soon as the monitor exists; it does not wait for the first run. Each later run (on the `period` schedule, or when started with Exa Trigger Search Monitor) happens on Exa's side, drops results earlier runs already returned, and posts the outcome to `webhook_url`. If Exa's response reports a `costDollars` total, the block bills it at 100 credits per USD; otherwise the call is free. Scheduled runs are not billed through this block, so it needs your own Exa API key and fails with a clear message on the platform's shared Exa key, which every user shares as one Exa account.

`webhook_url` must not be blank, and `period` must be a whole number followed by `h` or `d` (for example `6h`, `1d` or `7d`; mixed units like `1d12h` are rejected). The period defaults to `1d`; clear it to create a manual-only monitor. Content options that only `/search` supports, such as `snapshotAsOf` or a summary JSON schema, are dropped. The one-time `webhook_secret` is returned only here: paste it into the Exa Search Monitor Trigger block so deliveries are verified.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | What to watch for. Describe the ongoing signal rather than adding dates: each run only reports results that are new. | str | Yes |
| webhook_url | HTTPS URL Exa posts each run to. To handle runs in a workflow, add an 'Exa Search Monitor Trigger' block to it, save, and paste that block's webhook URL here. Must be publicly reachable (no localhost or private IPs); redirects are not followed. | str | Yes |
| period | How often the monitor runs: a whole number followed by h (hours) or d (days), minimum 1h, e.g. 6h, 1d or 7d. The schedule is anchored to the creation time and a run may start up to 30 minutes late. Leave empty for a manual-only monitor that runs only when triggered. | str | No |
| name | Optional display name for the monitor | str | No |
| num_results | Search results to fetch per run (1-100) | int | No |
| include_domains | Only return results from these domains | List[str] | No |
| exclude_domains | Never return results from these domains | List[str] | No |
| contents | Page content to attach to each result (text, highlights, summary, livecrawl and more) | ContentSettings | No |
| output_schema | Format of each run's synthesized output. Defaults to {"type": "text"} (a plain-text summary). Use {"type": "object", "properties": {...}} for JSON matching your schema; citations are returned separately in grounding, so leave them out of the schema. | Dict[str, Any] | No |
| metadata | Key-value pairs echoed back in every webhook delivery, useful for routing results | Dict[str, str] | No |
| webhook_events | Events to post to the webhook. Leave empty to receive all of them (lifecycle, run started and run completed). | List["monitor.created" \| "monitor.updated" \| "monitor.deleted" \| "monitor.run.created" \| "monitor.run.completed"] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| monitor_id | The new monitor's ID | str |
| webhook_secret | Secret for verifying this monitor's webhook deliveries. Exa shows it only once: paste it into the Exa Search Monitor Trigger block's webhook secret, or store it safely. | str |
| status | Monitor status, normally active | str |
| next_run_at | When the first scheduled run happens (ISO 8601); empty for a manual-only monitor | str |
| monitor | The full monitor | Monitor |

### Possible use case
<!-- MANUAL: use_case -->
**Competitor Watch**: Get a daily webhook with only the new pages announcing a competitor's product launches or pricing changes.

**Regulatory Tracking**: Watch government and agency domains for new rulings on a topic and route them to a review workflow using metadata.

**Market Signals**: Run a weekly monitor with an output schema that turns each run's new articles into structured fields such as company, location and investment size.
<!-- END MANUAL -->

---
