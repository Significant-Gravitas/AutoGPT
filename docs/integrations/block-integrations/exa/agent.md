# Exa Agent
<!-- MANUAL: file_description -->
Blocks for running Exa Agent, Exa's asynchronous research agent, to research questions, build lists and enrich records with cited answers.
<!-- END MANUAL -->

## Exa Agent Run

### What it is
Run Exa Agent, Exa's research agent, for deep research, list building and enrichment. Returns a cited answer, optionally as JSON matching your schema.

### How it works
<!-- MANUAL: how_it_works -->
This block starts a run on Exa's Agent API (`POST /agent/runs`), which plans and runs many searches, reads pages, verifies findings and returns a cited answer, as validated JSON in `structured` when you set an output schema. Before sending, it rejects an empty query, a max cost on a fixed-price effort (only auto and ultra accept one), a max duration on any effort other than ultra, and more than 5 data sources in total (built-in plus custom).

With wait for completion on (the default), the block polls the run every 5 seconds at first, stretching the gap up to 30 seconds, until it finishes or the polling timeout (default 900 seconds) passes. On timeout the block fails with the run ID and charges nothing; the run keeps going on Exa, so collect it with 'Wait for Exa Agent Run'. When the run finishes, the block charges the run's total cost as reported by Exa (the flat effort price, or metered usage for auto and ultra, plus any data source calls), unless another block has already charged you for that run; a failed run still returns its outputs and then fails the block with Exa's failure reason. With wait for completion off, it returns only the run ID, status and creation time and charges nothing; the run is charged once, later, by whichever block first sees it finished, such as 'Get Exa Agent Run' or 'Wait for Exa Agent Run'.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | What you want researched, built or enriched, in plain language. Be specific about criteria and how many results you want. | str | Yes |
| effort | How hard the agent works, which sets cost and speed. Fixed efforts have a flat price: minimal ($0.012, cheapest, very narrow lookups), low ($0.025, fast, simple lookups), medium ($0.10, balanced; the usual starting point), high ($0.50, slower, more thorough and better cited), xhigh ($1.00, the slowest fixed effort, for complex schemas and hard verification). Metered efforts bill actual usage up to a cap: auto (Exa picks the effort; up to $5 by default; good when the scope is unknown, such as list building) and ultra (the most exhaustive; up to $20 by default; usually ~30 minutes and up to 3 hours). | "minimal" \| "low" \| "medium" \| "high" \| "xhigh" \| "auto" \| "ultra" | No |
| output_schema | JSON Schema for the answer. When set, the result is returned as validated JSON in 'structured'. Make fields that may not be verifiable nullable, and bound lists with maxItems. | Dict[str, Any] | No |
| wait_for_completion | Wait for the run to finish and return its answer. Turn off to get the run ID immediately and collect it later with 'Wait for Exa Agent Run' or 'Get Exa Agent Run'. | bool | No |
| system_prompt | Extra instructions for how to research or answer, such as preferred sources or rules for what counts as a match | str | No |
| input_data | Records for the run to process or enrich, one object per record, e.g. [{"company": "Acme", "domain": "acme.com"}] | List[Dict[str, Any]] | No |
| exclusions | Records or entities the run must not return, e.g. ones you already have: [{"company": "Acme"}] | List[Dict[str, Any]] | No |
| previous_run_id | ID of an earlier run to follow up on (e.g. 'find 10 more'). The follow-up gets its own new run ID. | str | No |
| data_sources | Exa Connect partner databases the agent may query alongside web search (at most 5, billed per call on top of the run): fiber (Fiber.ai, B2B companies and people), similarweb (web traffic and rankings), baselayer (US business verification / KYB), polymarket (prediction market odds), macrobond (macroeconomic time series), affiliate (Affiliate.com product catalogs), particle (podcast transcripts), financial_datasets (US stock prices, fundamentals and filings), jinko (flight and hotel search). Mention the source in your query or output schema (e.g. 'monthly visits from Similarweb') so the agent uses it. | List["fiber" \| "similarweb" \| "baselayer" \| "polymarket" \| "macrobond" \| "affiliate" \| "particle" \| "financial_datasets" \| "jinko"] | No |
| custom_data_sources | Other Exa Connect provider IDs that Exa enabled for your team on request (e.g. harmonic). Counts toward the limit of 5. | List[str] | No |
| max_cost_dollars | Spending cap for this run in USD ($1-$100). Only for the auto and ultra efforts; Exa's default cap is $5 for auto and $20 for ultra. | float | No |
| max_duration_seconds | Soft time limit in seconds (300-10800, i.e. 5 minutes to 3 hours). Ultra effort only: near the limit the run wraps up and returns what it found. | int | No |
| metadata | Your own key-value labels stored with the run | Dict[str, str] | No |
| user_identifier | Opaque ID of the end user this run is for (max 64 characters), so Exa can attribute usage to them | str | No |
| polling_timeout | How long to wait for the run in seconds (10-10800). If it is still going, the block fails with the run ID so you can collect it later; the run keeps going on Exa. | int | No |

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
**Lead List Building**: Find companies matching criteria such as industry, funding stage and location, returned as a JSON list with founders and websites and cited sources.

**Record Enrichment**: Pass existing records in input data and have the run fill in missing fields such as headcount, domain or monthly visits from Similarweb.

**Deep Due Diligence**: Run an ultra-effort investigation of a company or market with a cost cap and a time limit, then follow up on the same run with previous run ID.
<!-- END MANUAL -->

---
