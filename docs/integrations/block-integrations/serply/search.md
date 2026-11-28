# Serply Search
<!-- MANUAL: file_description -->
Blocks for searching Google with the [Serply](https://serply.io) search API. See the [Serply docs](https://serply.io/docs) for API details.
<!-- END MANUAL -->

## Serply Search

### What it is
Searches Google using Serply's search API

### How it works
<!-- MANUAL: how_it_works -->
The block sends your query to Serply's `GET /v1/search` endpoint with your API key and returns the organic Google results, each with a title, URL, snippet and rank. Google operators such as `site:` and `filetype:` can be used inside the query, `country` localizes results (the `gl` parameter) and `time_range` limits results to the last day, week, month or year. The API returns at most 10 results per page, so the block always requests a full page and trims it to `max_results`; rows whose link is not an http(s) URL are skipped.

Beyond the raw `results` list (also emitted one `result` at a time), the block emits a `context` string with the results formatted as markdown, ready to feed straight into an LLM block. Each uncached search costs one Serply credit; cached responses are free.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The search query; supports Google operators such as site: and filetype: | str | Yes |
| max_results | Maximum number of results to return | int | No |
| country | Two-letter country code to localize results (e.g. us, gb, jp) | str | No |
| time_range | Only include results published within this time range | "day" \| "week" \| "month" \| "year" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the search failed | str |
| results | List of search results | List[SerplySearchResult] |
| result | Single search result | SerplySearchResult |
| context | A formatted string of the search results ready for LLMs. | str |

### Possible use case
<!-- MANUAL: use_case -->
Ground an agent's answer in fresh Google results, for example searching `site:docs.python.org asyncio` and passing `context` to an LLM block.
<!-- END MANUAL -->

---
