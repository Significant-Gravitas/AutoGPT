# Dataforb2B Reasoning
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Smart Search

### What it is
Natural-language search for people, leads or companies using DataForB2B's B2B database — describe your ideal lead or ICP in plain English and get matching profiles. Handles clarifying questions.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Plain-English LinkedIn search / ICP (e.g. 'marketing directors at Series A SaaS startups in France') | str | No |
| category | What to search for: 'people' or 'company' | str | No |
| session_id | Session id to resolve a previous 'needs_input' turn | str | No |
| answers | Answers to clarifying questions {question_id: answer} | Dict[str, Any] | No |
| max_results | Maximum results to return | int | No |
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
| applied_filters | The structured filters the search applied. Feed this into Linkedin People/Company Search 'filters_json' with an offset to paginate beyond the first page. | Dict[str, Any] |
| category | Category searched ('people' or 'companies') — route pagination to the matching search block | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
