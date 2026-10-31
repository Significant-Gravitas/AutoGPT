# Exa Agent Manage
<!-- MANUAL: file_description -->
Blocks for listing, cancelling, stopping and deleting Exa Agent runs and reading their stored events.
<!-- END MANUAL -->

## Exa Cancel Agent Run

### What it is
Cancel an Exa Agent run immediately. No results are kept; usage so far is still billed. To keep partial results of an ultra run, use 'Stop Exa Agent Run' instead.

### How it works
<!-- MANUAL: how_it_works -->
This block calls `POST /agent/runs/{id}/cancel`, which ends a queued or running run immediately and discards its results. It returns the run as Exa reports it after cancelling and, once Exa reports the run as finished, charges the usage the run built up before it was cancelled, unless another block has already charged you for that run. To keep what an ultra run has found so far, use 'Stop Exa Agent Run' instead.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| run_id | ID of the Exa Agent run to cancel | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| run_id | Exa Agent run ID (starts with agent_run_) | str |
| status | queued, running, completed, failed or cancelled | str |
| stop_reason | Why the run stopped: schema_satisfied (done), budget_reached, time_limit_reached, iteration_limit_reached, token_limit_reached, stopped (you stopped it; partial results kept), error or cancelled | str |
| text | The natural-language answer | str |
| structured | JSON matching your output schema (empty without a schema). Fields Exa could not support with evidence may be null. | Any |
| grounding | Citations for each output field, with confidence | List[AgentGrounding] |
| usage | Billed usage: compute units, searches, enrichments, data source calls | Usage |
| cost_total | Total cost of the run in USD | float |
| created_at | When the run was created (ISO 8601) | str |
| completed_at | When the run finished (ISO 8601), empty while running | str |

### Possible use case
<!-- MANUAL: use_case -->
**Cost Control**: Cancel a run that a workflow no longer needs, so it stops building up usage.

**Wrong Inputs**: Cancel a run started with a mistaken query or schema before it gets far.

**Workflow Abort**: Cancel outstanding runs when an earlier step of the workflow fails.
<!-- END MANUAL -->

---

## Exa Delete Agent Run

### What it is
Permanently delete a stored Exa Agent run and its output. It can no longer be fetched or used as a previous run.

### How it works
<!-- MANUAL: how_it_works -->
This block calls `DELETE /agent/runs/{id}` to permanently remove a stored run and its output, and returns the run ID with `deleted` true when Exa confirms. A deleted run can no longer be fetched or used as a previous run ID for a follow-up. Deleting cannot be undone.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| run_id | ID of the Exa Agent run to delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| run_id | ID of the deleted run | str |
| deleted | True when Exa deleted the run | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Data Hygiene**: Delete runs that processed sensitive input data once their results have been saved elsewhere.

**Retention Policy**: Remove old runs on a schedule to keep stored research within your retention rules.

**Clean Up Tests**: Delete runs created while building and testing a workflow.
<!-- END MANUAL -->

---

## Exa List Agent Run Events

### What it is
List an Exa Agent run's stored events (created, started, completed, failed, cancelled, plus progress). Not available for Zero Data Retention teams.

### How it works
<!-- MANUAL: how_it_works -->
This block calls `GET /agent/runs/{id}/events` and returns one page of the run's stored events in the order Exa stored them, such as `agent_run.created`, `agent_run.started` and `agent_run.completed`, plus progress events. Use `limit` (1-100, default 20) and pass `next_cursor` back as `cursor` while `has_more` is true to read further pages. Exa stores no events for Zero Data Retention teams, so the block fails for them.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| run_id | ID of the Exa Agent run | str | Yes |
| limit | How many events to return (1-100) | int | No |
| cursor | Cursor from a previous call's next_cursor, to fetch the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| events | Events in the order Exa stored them | List[AgentRunEvent] |
| event | Each event, one at a time | Event |
| has_more | Whether more events are available | bool |
| next_cursor | Pass as cursor to get the next page | str |

### Possible use case
<!-- MANUAL: use_case -->
**Progress Tracking**: Show how far a long ultra run has got by reading its latest events.

**Debugging Failures**: Read the `agent_run.failed` event to see the error code and message behind a failed run.

**Audit Trail**: Record when each run was created, started and finished for reporting.
<!-- END MANUAL -->

---

## Exa List Agent Runs

### What it is
List your Exa Agent runs, newest first, with paging

### How it works
<!-- MANUAL: how_it_works -->
This block calls `GET /agent/runs` and returns one page of your team's runs, newest first, both as a list and one at a time. Use `limit` (1-100, default 20) and pass `next_cursor` back as `cursor` while `has_more` is true to read further pages. Listing is free, and the list includes runs started by the legacy Research blocks. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account and would see each other's runs.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| limit | How many runs to return (1-100) | int | No |
| cursor | Cursor from a previous call's next_cursor, to fetch the next page | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| runs | Runs, newest first | List[AgentRun] |
| run | Each run, one at a time | Run |
| has_more | Whether more runs are available | bool |
| next_cursor | Pass as cursor to get the next page | str |

### Possible use case
<!-- MANUAL: use_case -->
**Run Dashboard**: List recent runs with their status, effort and cost for a team overview.

**Find a Run**: Look up the ID of an earlier run to fetch its result or follow up on it.

**Spend Review**: Page through runs and add up `cost` to track Exa Agent spending.
<!-- END MANUAL -->

---

## Exa Stop Agent Run

### What it is
Stop an Exa Agent run early and return the results gathered so far (stop reason 'stopped'). Exa currently supports this for ultra-effort runs only.

### How it works
<!-- MANUAL: how_it_works -->
This block calls `POST /agent/runs/{id}/stop`, which tells a running run to wrap up and keep the results gathered so far, with stop reason `stopped`. Exa currently supports this for ultra-effort runs only. The block returns the run as Exa reports it and, once Exa reports the run as finished, charges the run's total cost, unless another block has already charged you for that run; if the run is still wrapping up, collect the final result with 'Wait for Exa Agent Run', which charges it then.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| run_id | ID of the running Exa Agent run to stop | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| run_id | Exa Agent run ID (starts with agent_run_) | str |
| status | queued, running, completed, failed or cancelled | str |
| stop_reason | Why the run stopped: schema_satisfied (done), budget_reached, time_limit_reached, iteration_limit_reached, token_limit_reached, stopped (you stopped it; partial results kept), error or cancelled | str |
| text | The natural-language answer | str |
| structured | JSON matching your output schema (empty without a schema). Fields Exa could not support with evidence may be null. | Any |
| grounding | Citations for each output field, with confidence | List[AgentGrounding] |
| usage | Billed usage: compute units, searches, enrichments, data source calls | Usage |
| cost_total | Total cost of the run in USD | float |
| created_at | When the run was created (ISO 8601) | str |
| completed_at | When the run finished (ISO 8601), empty while running | str |

### Possible use case
<!-- MANUAL: use_case -->
**Good Enough Results**: Stop an ultra run once it has found enough and keep its partial answer.

**Deadline Cut-Off**: Stop a long run when a workflow's own deadline arrives instead of losing the results.

**Budget Guard**: Stop an ultra run that is close to the spend you are willing to put into it.
<!-- END MANUAL -->

---
