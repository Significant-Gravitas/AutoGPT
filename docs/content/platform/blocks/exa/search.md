# Exa Search

### What it is
Searches the web using Exa's advanced search API.

### What it does
Searches the web using Exa's advanced search API

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The search query | str | Yes |
| type | Type of search | "keyword" | "neural" | "fast" | No |
| category | Category to search within: company, research paper, news, pdf, github, tweet, personal site, linkedin profile, financial report | "company" | "research paper" | "news" | No |
| user_location | The two-letter ISO country code of the user (e.g., 'US') | str | No |
| number_of_results | Number of results to return | int | No |
| include_domains | Domains to include in search | List[str] | No |
| exclude_domains | Domains to exclude from search | List[str] | No |
| start_crawl_date | Start date for crawled content | str (date-time) | No |
| end_crawl_date | End date for crawled content | str (date-time) | No |
| start_published_date | Start date for published content | str (date-time) | No |
| end_published_date | End date for published content | str (date-time) | No |
| include_text | Text patterns to include | List[str] | No |
| exclude_text | Text patterns to exclude | List[str] | No |
| contents | Content retrieval settings | ContentSettings | No |
| moderation | Enable content moderation to filter unsafe content from search results | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| results | List of search results | List[ExaSearchResults] |
| result | Single search result | ExaSearchResults |
| context | A formatted string of the search results ready for LLMs. | str |
| search_type | For auto searches, indicates which search type was selected. | str |
| resolved_search_type | The search type that was actually used for this request (neural or keyword) | str |
| cost_dollars | Cost breakdown for the request | CostDollars |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
