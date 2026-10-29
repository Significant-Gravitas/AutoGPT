# Exa Research Queries
<!-- MANUAL: file_description -->
Blocks for checking on and waiting for Exa research tasks, which now run on Exa Agent because Exa retired its Research API.
<!-- END MANUAL -->

## Exa Get Research

### What it is
Get status and results of a research task. Legacy: now reads Exa Agent runs; use 'Get Exa Agent Run' for new workflows.

### How it works
<!-- MANUAL: how_it_works -->
Exa retired its Research API, so this block now reads the Exa Agent run with the given ID (`GET /agent/runs/{id}`) and returns it in the legacy research shape. Statuses map to pending (queued), running, completed, canceled (cancelled) and failed, and `model` is the legacy name for low, medium and high runs (`exa-research-fast`, `exa-research`, `exa-research-pro`) or the effort name for runs started with other efforts. Once the run has finished, it returns the total cost and the number of searches; Exa Agent reports no pages crawled or reasoning tokens, so `cost_pages` and `cost_reasoning_tokens` are always 0. `output_parsed` is only set when the run produced a JSON object, and for a failed run `error_message` gives Exa's reason.

Once the run has finished, the block charges its total cost, unless another block has already charged you for that run, so polling with it is safe. With include events on, it also reads up to 1,000 stored Exa Agent events (event, data and timestamp), which Exa does not keep for Zero Data Retention teams.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| research_id | The ID of the research task to retrieve (an Exa Agent run ID) | str | Yes |
| include_events | Include detailed event log of research operations | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| research_id | The research task identifier | str |
| status | Current status: pending, running, completed, canceled, or failed | str |
| instructions | The original research instructions | str |
| model | The research model used (or the Exa Agent effort, for runs started outside the Research blocks) | str |
| created_at | When research was created (Unix timestamp in ms) | int |
| finished_at | When research finished (Unix timestamp in ms, if completed/canceled/failed) | int |
| output_content | Research output as text (if completed) | str |
| output_parsed | Structured JSON output matching outputSchema (if provided and completed) | Dict[str, Any] |
| cost_total | Total cost in USD (if completed) | float |
| cost_searches | Number of searches performed (if completed) | int |
| cost_pages | Legacy: always 0, because Exa Agent does not report pages crawled | int |
| cost_reasoning_tokens | Legacy: always 0, because Exa Agent does not report reasoning tokens | int |
| error_message | Error message if research failed | str |
| events | Detailed event log (if include_events was True) | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
**Status Monitoring**: Check progress of long-running research tasks that were started without waiting.

**Result Retrieval**: Fetch finished research results from tasks started earlier in your workflow.

**Cost Tracking**: Review the total cost and searches of finished research for budgeting.
<!-- END MANUAL -->

---

## Exa Wait For Research

### What it is
Wait for a research task to complete with configurable timeout. Legacy: now waits on Exa Agent runs; use 'Wait for Exa Agent Run' for new workflows.

### How it works
<!-- MANUAL: how_it_works -->
Exa retired its Research API, so this block now polls the Exa Agent run with the given ID (`GET /agent/runs/{id}`) every `check_interval` seconds (default 10, fixed rather than growing) until it finishes or the timeout (default 600 seconds, up to 3600) passes. On timeout the block does not fail: it returns the last status with `timed_out` true, charges nothing and the run keeps going on Exa. Once the run finishes, the block charges the run's total cost as reported by Exa, unless another block has already charged you for that run, and returns the text answer, `output_parsed` (only when the run produced a JSON object) and the cost; `final_status` uses the legacy names pending, running, completed, canceled and failed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| research_id | The ID of the research task to wait for (an Exa Agent run ID) | str | Yes |
| timeout | Maximum time to wait in seconds | int | No |
| check_interval | Seconds between status checks | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| research_id | The research task identifier | str |
| final_status | Final status when polling stopped | str |
| output_content | Research output as text (if completed) | str |
| output_parsed | Structured JSON output (if outputSchema was provided and completed) | Dict[str, Any] |
| cost_total | Total cost in USD | float |
| elapsed_time | Total time waited in seconds | float |
| timed_out | Whether polling timed out before completion | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Sequential Workflows**: Make sure research finishes before dependent workflow steps run.

**Fire Then Collect**: Start research with wait for completion off, do other work, then wait for the result.

**Timeout Handling**: Use `timed_out` to handle research that takes longer than a time-sensitive workflow can wait.
<!-- END MANUAL -->

---
