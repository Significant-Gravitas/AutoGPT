# Stagehand Blocks
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Stagehand Act

### What it is
Interact with a web page by performing actions on a web page. Use it to build self-healing and deterministic automations that adapt to website chang.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| browserbase_project_id | Browserbase project ID (required if using Browserbase) | str | Yes |
| model | LLM to use for Stagehand (provider is inferred) | "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "claude-sonnet-4-5-20250929" \| "claude-sonnet-4-6" | No |
| url | URL to navigate to. | str | Yes |
| action | Action to perform. Suggested actions are: click, fill, type, press, scroll, select from dropdown. For multi-step actions, add an entry for each step. | List[str] | Yes |
| variables | Variables to use in the action. Variables contains data you want the action to use. | Dict[str, str] | No |
| dom_settle_timeout_ms | Timeout in ms to wait for the DOM to settle after navigation. | int | No |
| timeout_ms | Timeout in ms for each action. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the action was completed successfully | bool |
| message | Details about the action's execution. | str |
| action | Action performed | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Stagehand Extract

### What it is
Extract structured data from a webpage.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| browserbase_project_id | Browserbase project ID (required if using Browserbase) | str | Yes |
| model | LLM to use for Stagehand (provider is inferred) | "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "claude-sonnet-4-5-20250929" \| "claude-sonnet-4-6" | No |
| url | URL to navigate to. | str | Yes |
| instruction | Natural language description of elements or actions to discover. | str | Yes |
| dom_settle_timeout_ms | Timeout in ms to wait for the DOM to settle after navigation. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| extraction | Extracted data from the page. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Stagehand Observe

### What it is
Find suggested actions for your workflows

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| browserbase_project_id | Browserbase project ID (required if using Browserbase) | str | Yes |
| model | LLM to use for Stagehand (provider is inferred) | "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "claude-sonnet-4-5-20250929" \| "claude-sonnet-4-6" | No |
| url | URL to navigate to. | str | Yes |
| instruction | Natural language description of elements or actions to discover. | str | Yes |
| dom_settle_timeout_ms | Timeout in ms to wait for the DOM to settle after navigation. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| selector | XPath selector to locate element. | str |
| description | Human-readable description | str |
| method | Suggested action method | str |
| arguments | Additional action parameters | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
