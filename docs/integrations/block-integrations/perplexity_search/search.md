# Perplexity Search Search
<!-- MANUAL: file_description -->
Web search with the Perplexity Search API, which returns ranked results with titles, URLs and query-relevant snippets. Needs a Perplexity API key from console.perplexity.ai (`PERPLEXITY_API_KEY`). Perplexity charges $0.005 per web search request and $0.001 per fast search request.
<!-- END MANUAL -->

## Perplexity Search

### What it is
Search the web with the Perplexity Search API

### How it works
<!-- MANUAL: how_it_works -->
The block sends the query to `POST https://api.perplexity.ai/search` with Bearer API-key auth. `search_type` picks between `web` (the default, highest quality) and `fast` (quicker, lower price). `search_domain_filter` limits results to the listed domains, or excludes a domain when it is prefixed with `-`; `search_recency_filter` keeps only pages published within the last hour, day, week, month or year. Each row in `results` becomes a PerplexitySearchResult (title, url, snippet, and `date` or `last_updated` when Perplexity knows them). The block emits the full `results` list, then each `result` one at a time, then a `context` string with all results formatted for a prompt. That text is untrusted web content, so treat it as data, not instructions, before passing it to a model. HTTP failures, such as a bad API key (401), an invalid parameter (422) or a rate limit (429), raise a block error with Perplexity's message.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The search query | str | Yes |
| max_results | Maximum number of results to return (1 to 20) | int | No |
| search_type | Web search returns the highest quality results; fast search returns results sooner | "web" \| "fast" | No |
| search_domain_filter | Limit results to these domains (up to 20). Prefix a domain with '-' to exclude it instead. | List[str] | No |
| search_recency_filter | Only return pages published within this time window | "hour" \| "day" \| "week" \| "month" \| "year" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | List of search results | List[PerplexitySearchResult] |
| result | Each search result, emitted one at a time | PerplexitySearchResult |
| context | All results formatted as text for use in a prompt | str |

### Possible use case
<!-- MANUAL: use_case -->
**Research before writing**: Search for recent coverage of a topic with `search_recency_filter` set to `week`, then pass `context` to an AI text generator block to draft a summary with source URLs.

**Monitor specific sites**: Limit `search_domain_filter` to a few trusted domains and run the block on a schedule to pick up new pages, feeding each `result` into a notification block.
<!-- END MANUAL -->

---
