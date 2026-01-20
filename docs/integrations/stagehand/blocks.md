# Stagehand Blocks
<!-- MANUAL: file_description -->
Blocks for AI-powered browser automation using Stagehand and Browserbase.
<!-- END MANUAL -->

## Stagehand Act

### What it is
Interact with a web page by performing actions on a web page. Use it to build self-healing and deterministic automations that adapt to website chang.

### How it works
<!-- MANUAL: how_it_works -->
This block uses Stagehand with Browserbase to perform web actions using AI-powered element detection. Actions like click, fill, type, scroll, and select are described in natural language and executed reliably even if the page structure changes.

Configure timeouts for DOM settlement and page loading. Variables can be passed to actions for dynamic data entry.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| browserbase_project_id | Browserbase project ID (required if using Browserbase) | str | Yes |
| model | LLM to use for Stagehand (provider is inferred) | "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "claude-3-7-sonnet-20250219" | No |
| url | URL to navigate to. | str | Yes |
| action | Action to perform. Suggested actions are: click, fill, type, press, scroll, select from dropdown. For multi-step actions, add an entry for each step. | List[str] | Yes |
| variables | Variables to use in the action. Variables contains data you want the action to use. | Dict[str, str] | No |
| iframes | Whether to search within iframes. If True, Stagehand will search for actions within iframes. | bool | No |
| domSettleTimeoutMs | Timeout in milliseconds for DOM settlement.Wait longer for dynamic content | int | No |
| timeoutMs | Timeout in milliseconds for DOM ready. Extended timeout for slow-loading forms | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the action was completed successfully | bool |
| message | Details about the action’s execution. | str |
| action | Action performed | str |

### Possible use case
<!-- MANUAL: use_case -->
**Form Automation**: Fill out web forms, submit applications, or complete checkout flows.

**Self-Healing Scrapers**: Build automations that adapt to website changes without breaking.

**Testing Workflows**: Automate testing of web applications with resilient element targeting.
<!-- END MANUAL -->

---

## Stagehand Extract

### What it is
Extract structured data from a webpage.

### How it works
<!-- MANUAL: how_it_works -->
This block uses Stagehand with Browserbase to extract data from web pages using natural language instructions. Describe what data you want to extract, and the AI identifies and returns the matching content.

Supports searching within iframes and configurable timeouts for dynamic content that loads after the initial page render.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| browserbase_project_id | Browserbase project ID (required if using Browserbase) | str | Yes |
| model | LLM to use for Stagehand (provider is inferred) | "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "claude-3-7-sonnet-20250219" | No |
| url | URL to navigate to. | str | Yes |
| instruction | Natural language description of elements or actions to discover. | str | Yes |
| iframes | Whether to search within iframes. If True, Stagehand will search for actions within iframes. | bool | No |
| domSettleTimeoutMs | Timeout in milliseconds for DOM settlement.Wait longer for dynamic content | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| extraction | Extracted data from the page. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Data Scraping**: Extract product details, prices, or contact information from websites.

**Competitive Intelligence**: Pull data from competitor pages for analysis and monitoring.

**Research Automation**: Gather information from multiple web sources for research workflows.
<!-- END MANUAL -->

---

## Stagehand Observe

### What it is
Find suggested actions for your workflows

### How it works
<!-- MANUAL: how_it_works -->
This block analyzes a web page to discover available actions based on natural language instructions. It returns XPath selectors, action methods, and descriptions for elements matching your query.

Use this to explore a page's interactive elements before building automated workflows.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| browserbase_project_id | Browserbase project ID (required if using Browserbase) | str | Yes |
| model | LLM to use for Stagehand (provider is inferred) | "gpt-4.1-2025-04-14" \| "gpt-4.1-mini-2025-04-14" \| "claude-3-7-sonnet-20250219" | No |
| url | URL to navigate to. | str | Yes |
| instruction | Natural language description of elements or actions to discover. | str | Yes |
| iframes | Whether to search within iframes. If True, Stagehand will search for actions within iframes. | bool | No |
| domSettleTimeoutMs | Timeout in milliseconds for DOM settlement.Wait longer for dynamic content | int | No |

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
**Workflow Discovery**: Identify available actions on a page before building automations.

**Dynamic Navigation**: Discover clickable elements for pages with changing layouts.

**Automation Development**: Build robust automation workflows by understanding page structure.
<!-- END MANUAL -->

---
