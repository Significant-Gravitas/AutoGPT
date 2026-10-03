# Linkup Search
<!-- MANUAL: file_description -->
Blocks for searching the web in real time with Linkup.
<!-- END MANUAL -->

## Linkup Search

### What it is
Searches the web in real time using Linkup, returning relevant sources or a sourced answer

### How it works
<!-- MANUAL: how_it_works -->
The block sends your query to Linkup's search endpoint. With the default `searchResults` output type it returns relevant web sources, each with a title, URL and content extracted from the page, as a `results` list (also emitted one `result` at a time) plus a `context` string with the results formatted as markdown, ready to feed straight into an LLM block. With `sourcedAnswer` it instead returns a natural-language `answer` together with the `sources` that support it.

You can scope the search with domain include/exclude lists and a `from_date`/`to_date` publication window, and trade latency and cost for thoroughness with `depth`: `fast` or `standard` for most queries, `deep` for complex multi-step questions. The documented per-call price for the selected depth and output type is reported to the platform's cost tracking.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The search query | str | Yes |
| output_type | searchResults returns a list of relevant sources; sourcedAnswer returns an LLM-generated answer with the sources supporting it | "searchResults" \| "sourcedAnswer" | No |
| depth | Depth of the search: fast or standard for most queries, deep for complex multi-step questions (slower, 10x the cost) | "fast" \| "standard" \| "deep" | No |
| max_results | Maximum number of results to return | int | No |
| include_domains | Domains to restrict the search to | List[str] | No |
| exclude_domains | Domains to exclude from search | List[str] | No |
| from_date | Only include sources published on or after this date | str (date) | No |
| to_date | Only include sources published on or before this date | str (date) | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the search failed | str |
| results | List of search results (searchResults output type) | List[LinkupSearchResult] |
| result | Single search result | LinkupSearchResult |
| context | A formatted string of the search results ready for LLMs. | str |
| answer | LLM-generated answer to the query (sourcedAnswer output type) | str |
| sources | Sources supporting the answer (sourcedAnswer output type) | List[LinkupAnswerSource] |

### Possible use case
<!-- MANUAL: use_case -->
**Research Automation**: Pull current, citable sources on a topic and feed the `context` output directly into an LLM block for summarization or synthesis.

**Grounded Q&A**: Use the `sourcedAnswer` output type to get a concise answer with its supporting sources for chatbots or agents that need up-to-date facts.

**Focused Monitoring**: Restrict `include_domains` to trusted sites and set `from_date` to track recent developments from specific publications.
<!-- END MANUAL -->

---
