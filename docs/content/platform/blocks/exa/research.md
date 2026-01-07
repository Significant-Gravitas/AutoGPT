# Exa Create Research

### What it is
Create research task with optional waiting - explores web and synthesizes findings with citations.

### What it does
Create research task with optional waiting - explores web and synthesizes findings with citations

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| instructions | Research instructions - clearly define what information to find, how to conduct research, and desired output format. | str | Yes |
| model | Research model: 'fast' for quick results, 'standard' for balanced quality, 'pro' for thorough analysis | "exa-research-fast" | "exa-research" | "exa-research-pro" | No |
| output_schema | JSON Schema to enforce structured output. When provided, results are validated and returned as parsed JSON. | Dict[str, True] | No |
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
| output_parsed | Structured JSON output (only if wait_for_completion and outputSchema were provided) | Dict[str, True] |
| cost_total | Total cost in USD (only if wait_for_completion was True and completed) | float |
| elapsed_time | Time taken to complete in seconds (only if wait_for_completion was True) | float |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Get Research

### What it is
Get status and results of a research task.

### What it does
Get status and results of a research task

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| output_parsed | Structured JSON output matching outputSchema (if provided and completed) | Dict[str, True] |
| cost_total | Total cost in USD (if completed) | float |
| cost_searches | Number of searches performed (if completed) | int |
| cost_pages | Number of pages crawled (if completed) | int |
| cost_reasoning_tokens | AI tokens used for reasoning (if completed) | int |
| error_message | Error message if research failed | str |
| events | Detailed event log (if include_events was True) | List[Dict[str, True]] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa List Research

### What it is
List all research tasks with pagination support.

### What it does
List all research tasks with pagination support

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Wait For Research

### What it is
Wait for a research task to complete with configurable timeout.

### What it does
Wait for a research task to complete with configurable timeout

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| output_parsed | Structured JSON output (if outputSchema was provided and completed) | Dict[str, True] |
| cost_total | Total cost in USD | float |
| elapsed_time | Total time waited in seconds | float |
| timed_out | Whether polling timed out before completion | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
