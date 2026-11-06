# Exa Search Monitor Trigger
<!-- MANUAL: file_description -->
Blocks for starting a workflow when an Exa Search Monitor posts an event to its webhook.
<!-- END MANUAL -->

## Exa Search Monitor Trigger

### What it is
Triggers when an Exa Search Monitor reports new results. Paste this block's webhook URL into Exa Create Search Monitor, then paste the monitor's webhook secret here.

### How it works
<!-- MANUAL: how_it_works -->
This block gives you a webhook URL to paste into Exa Create Search Monitor; nothing is registered with Exa automatically. When `webhook_secret` is set, each delivery's `Exa-Signature` header (HMAC-SHA256 of the timestamp and body) is checked and unsigned or mismatched deliveries are rejected. Until it is set, any request to the URL is accepted. By default only `monitor.run.completed` starts the workflow; it carries the run's results, or why it failed, so check `status` and `fail_reason` before using the results.

For run events the block outputs the run's ID, status, new results (as a list and one at a time), synthesized content, grounding and fail reason. For monitor created, updated and deleted events it outputs the monitor instead. The monitor's metadata is output for every event.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webhook_secret | The webhook_secret output of Exa Create Search Monitor (Exa shows it only once). When set, deliveries without a valid Exa signature are rejected. Strongly recommended: until it is set, anyone who knows this block's URL can send it fake events. | str | No |
| events | Which monitor events start the workflow | Events | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| payload | The raw event Exa delivered | Dict[str, Any] |
| event_type | The event, e.g. monitor.run.completed | str |
| event_id | Unique ID of this event | str |
| created_at | When the event happened (ISO 8601) | str |
| monitor_id | The monitor the event is about | str |
| run_id | The run's ID (run events only) | str |
| status | Run status for run events (completed, failed, ...), or monitor status for monitor events | str |
| results | New search results found by the run | List[Dict[str, Any]] |
| result | Each result, one at a time | Dict[str, Any] |
| content | Synthesized findings: text, or JSON matching the monitor's output schema | Content |
| grounding | Citations and confidence for each synthesized field | List[MonitorGrounding] |
| fail_reason | Why the run failed; empty unless it failed | str |
| run | The full run | Run |
| monitor | The monitor (monitor created/updated/deleted events only) | Monitor |
| metadata | The monitor's metadata, echoed back by Exa | Dict[str, str] |

### Possible use case
<!-- MANUAL: use_case -->
**Alerts**: Send each run's new results to Slack or email as soon as the monitor finds them.

**Data Pipeline**: Append every new result to a spreadsheet or database for later analysis.

**Routing by Metadata**: Use the echoed metadata to send results from many monitors to the right team or client.
<!-- END MANUAL -->

---
