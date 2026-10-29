# Exa Research
<!-- MANUAL: file_description -->
Blocks for creating and listing Exa research tasks, which now run on Exa Agent because Exa retired its Research API.
<!-- END MANUAL -->

## Exa Create Research

### What it is
Create a research task with optional waiting. Legacy: now runs on Exa Agent (Exa retired its Research API); use the 'Exa Agent' block for new workflows.

### How it works
<!-- MANUAL: how_it_works -->
Exa retired its Research API, so this block now starts an Exa Agent run (`POST /agent/runs`) with your instructions as the query and your output schema, and the research ID it returns is the Exa Agent run ID. The legacy model picks a fixed Exa Agent effort: `exa-research-fast` runs at low ($0.025), `exa-research` (the default) at medium ($0.10) and `exa-research-pro` at high ($0.50). For other efforts, data sources, input data or cost caps, use the 'Exa Agent' block.

With wait for completion on, the block polls the run every 10 seconds until it finishes or the polling timeout (default 600 seconds, up to 3600) passes; on timeout the block fails and the run keeps going on Exa. Once the run finishes, the block charges the run's total cost as reported by Exa, unless another block has already charged you for that run, and returns the text answer and, when the run produced a JSON object, `output_parsed`. With wait for completion off, it returns the research ID, status `pending`, model, instructions and creation time and charges nothing; collect the run with 'Exa Wait For Research' or 'Exa Get Research', and whichever first sees it finished charges it.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| instructions | Research instructions - clearly define what information to find, how to conduct research, and desired output format. | str | Yes |
| model | Research depth, now mapped to an Exa Agent effort: 'exa-research-fast' runs at low effort ($0.025), 'exa-research' at medium ($0.10), 'exa-research-pro' at high ($0.50). The Exa Agent block offers every effort. | "exa-research-fast" \| "exa-research" \| "exa-research-pro" | No |
| output_schema | JSON Schema to enforce structured output. When provided, results are validated and returned as parsed JSON. | Dict[str, Any] | No |
| wait_for_completion | Wait for research to complete before returning. Ensures you get results immediately. | bool | No |
| polling_timeout | Maximum time to wait for completion in seconds (only if wait_for_completion is True) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| research_id | Unique identifier for tracking this research request (the Exa Agent run ID) | str |
| status | Final status of the research | str |
| model | The research model used | str |
| instructions | The research instructions provided | str |
| created_at | When the research was created (Unix timestamp in ms) | int |
| output_content | Research output as text (only if wait_for_completion was True and completed) | str |
| output_parsed | Structured JSON output (only if wait_for_completion and outputSchema were provided) | Dict[str, Any] |
| cost_total | Total cost in USD (only if wait_for_completion was True and completed) | float |
| elapsed_time | Time taken to complete in seconds (only if wait_for_completion was True) | float |

### Possible use case
<!-- MANUAL: use_case -->
**Market Research**: Research market trends, competitors or industry developments with cited sources.

**Due Diligence**: Gather background on companies, people or technologies and return it as structured JSON.

**Content Research**: Collect material on a topic for articles, reports or presentations with sources to cite.
<!-- END MANUAL -->

---

## Exa List Research

### What it is
List research tasks with pagination support. Legacy: now lists all of your team's Exa Agent runs; use 'List Exa Agent Runs' for new workflows.

### How it works
<!-- MANUAL: how_it_works -->
Exa retired its Research API, so this block now calls `GET /agent/runs` and returns one page (1-50, default 10) of your team's Exa Agent runs, newest first, in the legacy research task shape. The list includes every Exa Agent run, not only ones created by 'Exa Create Research': `model` shows the legacy model name for low, medium and high runs and the effort name (such as `auto` or `ultra`) for others, and statuses are reported as pending, running, completed, canceled or failed. Exa Agent reports no page or reasoning-token counts, so `num_pages` and `reasoning_tokens` in each task's cost are always 0. Listing is free, but it needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account and would see each other's runs.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| cursor | Cursor for pagination through results | str | No |
| limit | Number of research tasks to return (1-50) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| research_tasks | List of research tasks ordered by creation time (newest first) | List[ResearchTaskModel] |
| research_task | Individual research task (yielded for each task) | ResearchTaskModel |
| has_more | Whether there are more tasks to paginate through | bool |
| next_cursor | Cursor for the next page of results | str |

### Possible use case
<!-- MANUAL: use_case -->
**Research Management**: View active and finished research tasks for project management.

**Task Discovery**: Find earlier research tasks to retrieve their results or check their status.

**Activity Auditing**: Review research history and costs for compliance or reporting.
<!-- END MANUAL -->

---
