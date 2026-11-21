# SearchApi Search
<!-- MANUAL: file_description -->
Blocks for searching Google with the [SearchApi](https://www.searchapi.io) search API, which returns the live result page as structured JSON. See the [SearchApi docs](https://www.searchapi.io/docs/google) for API details.
<!-- END MANUAL -->

## SearchApi Search

### What it is
Searches Google using SearchApi's search API

### How it works
<!-- MANUAL: how_it_works -->
The block sends your query to SearchApi's `GET /api/v1/search` endpoint with `engine=google` and your API key (a `Bearer` token) and returns the organic Google results, each with a title, URL, snippet and rank. Google operators such as `site:` and `filetype:` can be used inside the query, `country` localizes results (the `gl` parameter) and `time_range` limits results to the last hour, day, week, month or year (the `time_period` parameter). Google returns at most 10 results per page, so `max_results` is capped at 10; the block asks for that many (`num`) and trims the page to the limit. Rows whose link is not an http(s) URL are skipped; links that Google wraps in a `google.com/goto?url=` redirect are passed through as they are. A query with no hits is an empty `results` list, not an error.

Beyond the raw `results` list (also emitted one `result` at a time), the block emits a `context` string with the results formatted as markdown, ready to feed straight into an LLM block. Each successful search costs one SearchApi search; failed requests are not billed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The search query; supports Google operators such as site: and filetype: | str | Yes |
| max_results | Maximum number of results to return | int | No |
| country | Two-letter country code to localize results (e.g. us, gb, jp) | str | No |
| time_range | Only include results published within this time range | "hour" \| "day" \| "week" \| "month" \| "year" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the search failed | str |
| results | List of search results | List[SearchApiSearchResult] |
| result | Single search result | SearchApiSearchResult |
| context | A formatted string of the search results ready for LLMs. | str |

### Possible use case
<!-- MANUAL: use_case -->
Ground an answer in fresh Google results, for example searching `site:docs.python.org asyncio` and passing `context` to an LLM block, or watch what Google shows for your brand in a given country by setting `country` and `time_range`.
<!-- END MANUAL -->

---
