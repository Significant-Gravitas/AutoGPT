# Exa Agent Lookup
<!-- MANUAL: file_description -->
Blocks for checking on and waiting for Exa Agent runs started elsewhere in a workflow.
<!-- END MANUAL -->

## Exa Get Agent Run

### What it is
Get an Exa Agent run's status and, once it has finished, its answer, structured output, citations and cost

### How it works
<!-- MANUAL: how_it_works -->
This block reads one run with `GET /agent/runs/{id}` and returns its current status and, once it has finished, the answer, structured output, citations, usage and cost, plus the query and effort it was given. Once the run has finished, the block charges its total cost as reported by Exa, unless another block has already charged you for that run: each run is billed once, so polling with this block is safe. A failed run does not fail the block; instead `failure_reason` explains why, read from the run or, failing that, from its stored events.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| run_id | ID of the Exa Agent run | str | Yes |

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
| query | The query the run was given | str |
| effort | The effort the run used | str |
| failure_reason | Why the run failed (only when status is failed) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Status Check**: Look up a run started with wait for completion off and branch on whether it is still queued, running or finished.

**Result Retrieval**: Fetch the answer and citations of a finished run by its ID, for example in a later execution of the workflow.

**Failure Handling**: Read `failure_reason` from a failed run and route the workflow to a retry or a notification.
<!-- END MANUAL -->

---

## Exa Wait For Agent Run

### What it is
Wait for an Exa Agent run to finish, then return its answer, structured output, citations and cost

### How it works
<!-- MANUAL: how_it_works -->
This block polls `GET /agent/runs/{id}` until the run is completed, failed or cancelled, or the timeout (default 900 seconds, up to 10800) passes. The first re-check comes after `check_interval` seconds (default 5) and each gap after that grows by half, up to 30 seconds, so short runs return quickly and long ultra runs are not polled constantly. On timeout the block does not fail: it returns the run's current state with `timed_out` true and charges nothing, and the run keeps going on Exa.

When the run has finished, the block charges the run's total cost as reported by Exa, unless another block has already charged you for that run; each run is billed once, whichever block sees it finish first. A failed run does not fail the block; `failure_reason` explains why it failed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| run_id | ID of the Exa Agent run to wait for | str | Yes |
| timeout | Longest time to wait in seconds (1-10800). Ultra runs usually take about 30 minutes and can take up to 3 hours. | int | No |
| check_interval | Seconds before the first re-check (1-60); the gap then grows gradually up to 30 seconds | int | No |

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
| timed_out | True if the run was still going when the timeout passed | bool |
| elapsed_seconds | How long the block waited | float |
| failure_reason | Why the run failed (only when status is failed) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Fire Then Collect**: Start several Exa Agent runs with wait for completion off, do other work, then wait for each run's result.

**Long Ultra Runs**: Wait up to 3 hours for an ultra-effort run without holding up the block that started it.

**Timeout Fallback**: Use `timed_out` to send a partial-progress notice or stop the run when it takes longer than the workflow can wait.
<!-- END MANUAL -->

---
