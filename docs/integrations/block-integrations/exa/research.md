# Exa Research
<!-- MANUAL: file_description -->
Blocks for creating and managing autonomous research tasks using Exa's Research API.
<!-- END MANUAL -->

## Exa Create Research

### What it is
Create research task with optional waiting - explores web and synthesizes findings with citations

### How it works
<!-- MANUAL: how_it_works -->
This block creates an asynchronous research task using Exa's Research API. The API autonomously explores the web, searches for relevant information, and synthesizes findings into a comprehensive report with citations.

You can choose from different model tiers (fast, standard, pro) depending on your speed vs. depth requirements. The block supports structured output via JSON Schema and can optionally wait for completion to return results immediately.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| instructions | Research instructions - clearly define what information to find, how to conduct research, and desired output format. | str | Yes |
| model | Research model: 'fast' for quick results, 'standard' for balanced quality, 'pro' for thorough analysis | "exa-research-fast" \| "exa-research" \| "exa-research-pro" | No |
| output_schema | JSON Schema to enforce structured output. When provided, results are validated and returned as parsed JSON. | Dict[str, Any] | No |
| wait_for_completion | Wait for research to complete before returning. Ensures you get results immediately. | bool | No |
| polling_timeout | Maximum time to wait for completion in seconds (only if wait_for_completion is True) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| research_id | Unique identifier for tracking this research request | str |
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
**Market Research**: Automatically research market trends, competitors, or industry developments with cited sources.

**Due Diligence**: Conduct comprehensive background research on companies, people, or technologies.

**Content Research**: Gather research on topics for articles, reports, or presentations with proper citations.
<!-- END MANUAL -->

---

## Exa Get Research

### What it is
Get status and results of a research task

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves the current status and results of a previously created research task. You can check whether the research is still running, completed, or failed.

When the research is complete, the block returns the full output content along with cost breakdown including searches performed, pages crawled, and tokens used. You can also optionally retrieve the detailed event log of research operations.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| research_id | The ID of the research task to retrieve | str | Yes |
| include_events | Include detailed event log of research operations | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| research_id | The research task identifier | str |
| status | Current status: pending, running, completed, canceled, or failed | str |
| instructions | The original research instructions | str |
| model | The research model used | str |
| created_at | When research was created (Unix timestamp in ms) | int |
| finished_at | When research finished (Unix timestamp in ms, if completed/canceled/failed) | int |
| output_content | Research output as text (if completed) | str |
| output_parsed | Structured JSON output matching outputSchema (if provided and completed) | Dict[str, Any] |
| cost_total | Total cost in USD (if completed) | float |
| cost_searches | Number of searches performed (if completed) | int |
| cost_pages | Number of pages crawled (if completed) | int |
| cost_reasoning_tokens | AI tokens used for reasoning (if completed) | int |
| error_message | Error message if research failed | str |
| events | Detailed event log (if include_events was True) | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
**Status Monitoring**: Check progress of long-running research tasks that were started asynchronously.

**Result Retrieval**: Fetch completed research results from tasks started earlier in your workflow.

**Cost Tracking**: Review the cost breakdown of completed research for budgeting and optimization.
<!-- END MANUAL -->

---

## Exa List Research

### What it is
List all research tasks with pagination support

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves a list of all your research tasks, ordered by creation time with newest first. It supports pagination for handling large numbers of tasks.

The block returns basic information about each task including its ID, status, instructions, and timestamps. Use this to find specific research tasks or monitor all ongoing research activities.
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
**Research Management**: View all active and completed research tasks for project management.

**Task Discovery**: Find previously created research tasks to retrieve their results or check status.

**Activity Auditing**: Review research activity history for compliance or reporting purposes.
<!-- END MANUAL -->

---

## Exa Wait For Research

### What it is
Wait for a research task to complete with configurable timeout

### How it works
<!-- MANUAL: how_it_works -->
This block polls a research task until it completes or times out. It periodically checks the task status at configurable intervals and returns the final results when done.

The block is useful when you need to block workflow execution until research completes. It returns whether the operation timed out, allowing you to handle incomplete research gracefully.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| research_id | The ID of the research task to wait for | str | Yes |
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
**Sequential Workflows**: Ensure research completes before proceeding to dependent workflow steps.

**Synchronous Integration**: Convert asynchronous research into synchronous operations for simpler workflow logic.

**Timeout Handling**: Implement research with graceful timeout handling for time-sensitive applications.
<!-- END MANUAL -->

---
