# Tavily Search
<!-- MANUAL: file_description -->
Blocks for searching the web with Tavily's AI-native search API.
<!-- END MANUAL -->

## Tavily Search

### What it is
Searches the web using Tavily's AI-native search API

### How it works
<!-- MANUAL: how_it_works -->
The block sends your query to Tavily's search endpoint and returns ranked web results, each with a relevance score and a query-relevant content snippet. You can scope the search by `topic` (general, news, or finance), `time_range`, and domain include/exclude lists, and trade cost for quality with `search_depth` (basic/fast/ultra-fast at 1 credit, advanced at 2).

Beyond the raw `results` list (also emitted one `result` at a time), the block can return an LLM-generated `answer` synthesized from the results when `include_answer` is enabled, and always emits a `context` string — the results formatted as markdown, ready to feed straight into an LLM block. Actual credit spend is read from the API's usage report and reported to the platform's cost tracking.
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
**Research Automation**: Pull current, ranked sources on a topic and feed the `context` output directly into an LLM block for summarization or synthesis.

**Grounded Q&A**: Enable `include_answer` to get a concise, source-backed answer for chatbots or agents that need up-to-date facts.

**News & Market Monitoring**: Set `topic` to news or finance and narrow `time_range` to recent windows to track breaking developments.
<!-- END MANUAL -->

---
