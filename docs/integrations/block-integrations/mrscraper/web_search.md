# MrScraper Web Search
<!-- MANUAL: file_description -->
Searches the web and returns ranked results. Needs a MrScraper API key: create one at [app.mrscraper.com/api-tokens](https://app.mrscraper.com/api-tokens) and add it as a MrScraper credential.
<!-- END MANUAL -->

## MrScraper Web Search

### What it is
Search the web and return ranked results with titles, URLs, descriptions, and sources.

### How it works
<!-- MANUAL: how_it_works -->
The block sends `GET https://api.mrscraper.com/` with `action=websearch`, the query and `numResults`, with a fixed 120-second deadline. The query is trimmed and must have 1 to 1000 characters. `country` is sent in lowercase and left out when empty. The block outputs the list as `results`, then each item as `result` in rank order, and the tokens used as `token_usage`. Fewer results than requested may come back. Titles and descriptions are untrusted web text.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query text, trimmed before sending, containing 1 to 1000 characters. | str | Yes |
| country | Two-letter country code to search from, sent in lowercase. | str | No |
| num_results | Number of search results to request, as an integer from 10 to 100. Fewer results may be returned. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | Ranked search results. | List[MrScraperSearchResult] |
| result | One search result, emitted for each result in order. | MrScraperSearchResult |
| token_usage | MrScraper tokens used by the search. | int |

### Possible use case
<!-- MANUAL: use_case -->
**Find pages, then read them**: Connect `result.url` to MrScraper Scrape Markdown to read each hit in full.

**Local results**: Set `country` to search as a user in that country would.

**Monitor mentions**: Run a `site:` or brand query on a schedule and send new results to a channel.
<!-- END MANUAL -->

---
