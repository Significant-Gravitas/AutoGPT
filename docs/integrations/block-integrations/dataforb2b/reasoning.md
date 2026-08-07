# DataForB2B Reasoning
<!-- MANUAL: file_description -->
Natural-language, LLM-friendly search over DataForB2B's B2B database — describe the people or companies you're looking for in plain English instead of building structured filters by hand.
<!-- END MANUAL -->

## Smart Search

### What it is
Natural-language search for people, leads or companies using DataForB2B's B2B database — describe your ideal lead or ICP in plain English and get matching profiles. Handles clarifying questions.

### How it works
<!-- MANUAL: how_it_works -->
The block sends either a natural-language `query` or a continuation pair of `session_id` and `answers` to DataForB2B's reasoning-search endpoint; mixed or incomplete request modes are rejected. The API translates the query into structured filters and may return `status: needs_input` with `questions` and a `session_id` for a follow-up call. On success, copy `applied_filters` into People Search or Company Search's `filters_json` input and set `offset` to paginate beyond the first page; an `ok` response may still contain an empty `results` list. Client and server errors are caught and surfaced via the `error` output instead of raising an exception.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Plain-English LinkedIn search / ICP (e.g. 'marketing directors at Series A SaaS startups in France') | str | No |
| category | What to search for: 'people' or 'company' | "people" \| "company" | No |
| session_id | Session id to resolve a previous 'needs_input' turn | str | No |
| answers | Answers to clarifying questions {question_id: answer} | Dict[str, Any] | No |
| max_results | Maximum results to return (1-100) | int | No |
| enrich_live | Fetch fresh live data (uses more credits) | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the search failed | str |
| result | Full reasoning-search response | Dict[str, Any] |
| status | 'ok' or 'needs_input' | str |
| results | Matching results when status is ok | List[Any] |
| questions | Clarifying questions when status is needs_input | List[Any] |
| session_id | Session id to continue the search | str |
| applied_filters | The structured filters the search applied. Feed this into People Search or Company Search 'filters_json' with an offset to paginate beyond the first page. | Dict[str, Any] |
| category | Category searched ('people' or 'company', echoed from the input) — route pagination to the matching search block | str |

### Possible use case
<!-- MANUAL: use_case -->
**Conversational Lead Sourcing**: Let an agent describe an ICP in plain English (e.g. "marketing directors at Series A SaaS startups in France") and resolve it into a structured search without manually building filters.

**Multi-Turn Refinement**: Handle ambiguous requests by answering clarifying questions returned by the API, then continue the same session to completion.

**Search Handoff**: Reuse `applied_filters` in People Search or Company Search to paginate a natural-language result set.
<!-- END MANUAL -->

---
