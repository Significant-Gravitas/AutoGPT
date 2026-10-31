# Exa Search Monitor Runs
<!-- MANUAL: file_description -->
Blocks for starting Exa Search Monitor runs and reading their status and results.
<!-- END MANUAL -->

## Exa Get Search Monitor Run

### What it is
Get an Exa Search Monitor run's status, new results and synthesized findings

### How it works
<!-- MANUAL: how_it_works -->
This block sends `GET /monitors/{monitor_id}/runs/{run_id}` to Exa once; it does not poll. A run's status is `pending`, `running`, `completed`, `failed` or `cancelled`, and its results, synthesized `content` and `grounding` stay empty until it completes. `content` is plain text, or JSON when the monitor has an object output schema, and `fail_reason` explains failures such as `insufficient_credits`. The block also outputs each new result one at a time. If Exa's response reports a `costDollars` total, the block bills it at 100 credits per USD; otherwise the call is free. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| monitor_id | The monitor the run belongs to | str | Yes |
| run_id | The run to fetch | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| run_id | The run's ID | str |
| status | pending, running, completed, failed or cancelled | str |
| results | New search results the run found (empty until completed) | List[Dict[str, Any]] |
| result | Each result, one at a time | Dict[str, Any] |
| content | Synthesized findings: text, or JSON matching the monitor's output schema | Content |
| grounding | Citations and confidence for each synthesized field | List[MonitorGrounding] |
| fail_reason | Why the run failed, e.g. insufficient_credits or source_not_available; empty unless failed | str |
| run | The full run | Run |

### Possible use case
<!-- MANUAL: use_case -->
**Result Retrieval**: Fetch the findings of a run whose ID arrived from the Exa Search Monitor Trigger block or a list of runs.

**Failure Diagnosis**: Read `fail_reason` for a failed run to see whether credits ran out or a source was unavailable.

**Citation Review**: Use `grounding` to show which sources back each field of a run's structured output.
<!-- END MANUAL -->

---

## Exa List Search Monitor Runs

### What it is
List an Exa Search Monitor's runs, newest first, with status and output

### How it works
<!-- MANUAL: how_it_works -->
This block sends `GET /monitors/{monitor_id}/runs` to Exa and returns one page of runs, newest first, as a list and one at a time. Output (results, content and grounding) is filled in only for completed runs. When `has_more` is true, pass `next_cursor` back as `cursor` to get older runs. The block does not charge for the call. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| monitor_id | The monitor whose runs to list | str | Yes |
| limit | Runs per page (1-100) | int | No |
| cursor | Cursor from a previous page's next_cursor | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| runs | Runs on this page, newest first. Output is filled in only for completed runs. | List[SearchMonitorRunModel] |
| run | Each run, one at a time | Run |
| has_more | Whether another page exists | bool |
| next_cursor | Pass as cursor to fetch the next page | str |

### Possible use case
<!-- MANUAL: use_case -->
**Run History**: Show a monitor's recent runs and their statuses in a report.

**Catch-up**: Collect results from runs whose webhook deliveries a workflow missed.

**Reliability Tracking**: Count failed runs over time to spot a monitor that needs attention.
<!-- END MANUAL -->

---

## Exa Trigger Search Monitor

### What it is
Run an Exa Search Monitor now; results go to its webhook and can also be read with List/Get Search Monitor Runs

### How it works
<!-- MANUAL: how_it_works -->
This block sends `POST /monitors/{monitor_id}/trigger` to Exa to start a run straight away, outside the schedule; it works for active and paused monitors. It returns once Exa accepts the request, with `triggered` reporting whether a run started, and does not wait for the run to finish. The results go to the monitor's webhook and can also be read with List or Get Search Monitor Runs. If Exa's response reports a `costDollars` total, the block bills it at 100 credits per USD; otherwise the call is free. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| monitor_id | The monitor to run. Works for active and paused monitors. | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| monitor_id | The monitor that was triggered | str |
| triggered | Whether Exa started the run | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Manual-Only Monitors**: Run a monitor created without a schedule whenever a workflow decides it is needed.

**First Results Now**: Trigger a new monitor immediately instead of waiting for its first scheduled run.

**Paused Monitors**: Run a paused monitor once on demand without putting it back on its schedule.
<!-- END MANUAL -->

---
