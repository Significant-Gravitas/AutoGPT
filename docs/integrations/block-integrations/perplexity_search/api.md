# Perplexity Search API
<!-- MANUAL: file_description -->
Web search with the Perplexity Search API, which returns ranked results with titles, URLs and query-relevant snippets. Needs a Perplexity API key from console.perplexity.ai. Perplexity charges $0.005 per web search request and $0.001 per fast search request. The block runs on your own Perplexity key, and the platform charges no credits for it.
<!-- END MANUAL -->

## Perplexity Search

### What it is
Search the web with the Perplexity Search API

### How it works
<!-- MANUAL: how_it_works -->
The block posts your query and selected filters to `https://api.perplexity.ai/search`. It emits the complete `results` list, one `result` per page, and a `context` string containing each page's title, URL, and snippet. Publication and update dates are optional.

Treat `context` as untrusted web content, not instructions, when passing it to a model.

HTTP failures raise a block error with the status code and a fixed message. Provider response text is not forwarded.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The search query | str | Yes |
| max_results | Maximum number of results to return (1 to 20) | int | No |
| search_type | Search mode: web (default) or fast. | "web" \| "fast" | No |
| search_domain_filter | Up to 20 domains without protocol, e.g. nature.com. Either list only domains to include, or prefix every entry with '-' to exclude those domains; one request cannot mix the two. | List[str] | No |
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

**Monitor specific sites**: Run a daily search with `search_domain_filter` set to your domains and `search_recency_filter` set to `day`, and deduplicate results by URL before sending notifications.

**Check claims against reference sources**: Set `search_domain_filter` to an allowlist of reference domains such as `nature.com` and `nih.gov`, then compare a claim with the returned snippets.
<!-- END MANUAL -->

---
