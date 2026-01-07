# Exa Contents

### What it is
Retrieves document contents using Exa's contents API.

### What it does
Retrieves document contents using Exa's contents API

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| urls | Array of URLs to crawl (preferred over 'ids') | List[str] | No |
| ids | [DEPRECATED - use 'urls' instead] Array of document IDs obtained from searches | List[str] | No |
| text | Retrieve text content from pages | bool | No |
| highlights | Text snippets most relevant from each page | HighlightSettings | No |
| summary | LLM-generated summary of the webpage | SummarySettings | No |
| livecrawl | Livecrawling options: never, fallback (default), always, preferred | "never" | "fallback" | "always" | No |
| livecrawl_timeout | Timeout for livecrawling in milliseconds | int | No |
| subpages | Number of subpages to crawl | int | No |
| subpage_target | Keyword(s) to find specific subpages of search results | str | List[str] | No |
| extras | Extra parameters for additional content | ExtrasSettings | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| results | List of document contents with metadata | List[ExaSearchResults] |
| result | Single document content result | ExaSearchResults |
| context | A formatted string of the results ready for LLMs | str |
| request_id | Unique identifier for the request | str |
| statuses | Status information for each requested URL | List[ContentStatus] |
| cost_dollars | Cost breakdown for the request | CostDollars |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
