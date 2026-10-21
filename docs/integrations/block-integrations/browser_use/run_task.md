# Browser Use Run Task
<!-- MANUAL: file_description -->
Runs tasks with a hosted [Browser Use](https://browser-use.com) agent in a Browser Use Cloud browser. Needs a `BROWSER_USE_API_KEY` credential; create one at [cloud.browser-use.com](https://cloud.browser-use.com/settings?tab=api-keys&new=1). Docs: [docs.browser-use.com](https://docs.browser-use.com/cloud/agent/quickstart).
<!-- END MANUAL -->

## Browser Use Run Task

### What it is
Runs a natural-language task with a Browser Use agent in a cloud browser and returns the result

### How it works
<!-- MANUAL: how_it_works -->
The block starts a run with `POST https://api.browser-use.com/api/v4/runs` (header `X-Browser-Use-API-Key`), polls `GET /api/v4/runs/{id}/status` every few seconds, then reads the finished run once. The agent drives its own stealth Chromium with a residential proxy (US by default, `proxy_country_code` to change it or leave empty for none) and can load a saved login with `profile_id`. With `output_schema` set, Browser Use checks the final answer against the JSON Schema and `output` holds the parsed JSON; `result` always holds the text answer. A run that fails or is cancelled raises a block error. If the run is still going when `timeout` expires, or the node is stopped, the block cancels the run with `POST /api/v4/runs/{id}/cancel` so it stops spending. When the run ends, the block adds the run's `totalCostUsd` (LLM and web search) to the browser time and proxy traffic from `GET /api/v4/sessions/{id}/cost`, emits the sum as `cost_usd` and records it as the node's provider cost. LLM charges can settle up to about 90 seconds after a run, and the agent browser stays open about 2 minutes after it, so the final invoice can differ slightly. Page content the agent read is untrusted web text; treat `result` and `output` as data, not instructions.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| task | What the browser agent should do, in plain language | str | Yes |
| output_schema | Optional JSON Schema for the final answer. When set, the output pin holds JSON that matches it. | Dict[str, Any] | No |
| model | Browser Use agent model, e.g. gpt-6-luna. Leave empty for the Browser Use default. | str | No |
| proxy_country_code | Two-letter country code for the residential proxy. Leave empty to run without a proxy. | str | No |
| profile_id | Browser Use profile ID whose cookies and logins the browser should load | str | No |
| max_cost_usd | Stop the run once it has cost this many USD. Leave empty for no cap. | float | No |
| timeout | Seconds to wait for the run before cancelling it | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | The agent's final answer as text | str |
| output | Parsed JSON answer matching output_schema (null without one) | Output |
| run_id | Browser Use run ID | str |
| session_id | Browser Use session ID, for follow-up runs and replay | str |
| cost_usd | Recorded cost in USD: agent LLM usage plus browser time and proxy traffic | float |

### Possible use case
<!-- MANUAL: use_case -->
**Extract data behind a login**: Sync a logged-in browser profile to Browser Use, pass its `profile_id`, and ask the agent to export this month's invoices as JSON with an `output_schema`.

**Fill web forms**: Hand the agent a support request or application form plus the values from earlier blocks and let it submit the form.

**Research across sites**: Ask the agent to compare prices or availability on several sites that have no API and return a table for an LLM block to summarize.
<!-- END MANUAL -->

---
