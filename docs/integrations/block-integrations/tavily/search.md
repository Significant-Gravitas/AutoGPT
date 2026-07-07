# Tavily Search
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Tavily Search

### What it is
Searches the web using Tavily's AI-native search API

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The search query | str | Yes |
| topic | Search category: general, news, or finance | "general" \| "news" \| "finance" | No |
| search_depth | Depth of the search: basic, fast or ultra-fast (1 credit), or advanced (2 credits) | "basic" \| "advanced" \| "fast" \| "ultra-fast" | No |
| max_results | Maximum number of results to return | int | No |
| time_range | Only include results published within this time range | "day" \| "week" \| "month" \| "year" | No |
| include_domains | Domains to include in search | List[str] | No |
| exclude_domains | Domains to exclude from search | List[str] | No |
| include_answer | Include an LLM-generated answer to the query, based on the search results | bool | No |
| include_raw_content | Include the full page content for each result | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the search failed | str |
| results | List of search results | List[TavilySearchResult] |
| result | Single search result | TavilySearchResult |
| answer | LLM-generated answer to the query, based on the search results | str |
| context | A formatted string of the search results ready for LLMs. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
