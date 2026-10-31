# Exa Websets Monitor Runs
<!-- MANUAL: file_description -->
Blocks for checking the scheduled runs of Exa Websets monitors.
<!-- END MANUAL -->

## Exa Get Webset Monitor Run

### What it is
Get the status and outcome of one scheduled run of a Websets monitor

### How it works
<!-- MANUAL: how_it_works -->
This block reads one run from the Websets API's `/v0/monitors/{monitor_id}/runs/{run_id}` and returns it once, without waiting or polling. Status is one of created, running, completed, canceled or failed, and type says whether the run was a search or a refresh.

`failed_reason` is filled only for failed runs and `completed_at` only once the run has finished. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| monitor_id | The ID of the Websets monitor the run belongs to | str | Yes |
| run_id | The ID of the run to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| run | The full run details | Run |
| run_id | The unique identifier for the run | str |
| status | created, running, completed, canceled or failed | str |
| type | What the run did: search or refresh | str |
| failed_reason | Why the run failed (only for failed runs) | str |
| completed_at | When the run completed, if it has | str |
| created_at | When the run started | str |

### Possible use case
<!-- MANUAL: use_case -->
**Run Health Check**: Fetch the latest run of a monitor and alert someone if it failed.

**Failure Diagnosis**: Read the failed reason of a run to decide whether to fix the monitor or retry.

**Completion Gate**: Continue a workflow only after a specific monitor run has completed.
<!-- END MANUAL -->

---

## Exa List Webset Monitor Runs

### What it is
List the scheduled runs of a Websets monitor, to check whether its updates succeeded

### How it works
<!-- MANUAL: how_it_works -->
This block lists a monitor's runs from the Websets API's `/v0/monitors/{monitor_id}/runs` in a single request. It outputs the whole list and then each run one at a time, plus `has_more` and `next_cursor` as Exa returns them.

The block has no limit or cursor input, so it always returns Exa's first page; `next_cursor` can't be passed back to it. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| monitor_id | The ID of the Websets monitor whose runs to list | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| runs | The monitor's runs | List[WebsetMonitorRunModel] |
| run | Individual run (yielded for each run in the list) | Run |
| has_more | Whether Exa has more runs than it returned | bool |
| next_cursor | Cursor Exa returned for the next page, if any | str |

### Possible use case
<!-- MANUAL: use_case -->
**Monitor Audit**: Review recent runs of a monitor to see how often its scheduled updates succeed.

**Failure Alerts**: Loop over runs and flag any with a failed status.

**Activity Reporting**: Count search and refresh runs for a periodic report on a monitor.
<!-- END MANUAL -->

---
