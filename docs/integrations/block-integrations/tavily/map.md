# Tavily Map
<!-- MANUAL: file_description -->
Blocks for discovering a website's URL structure with Tavily.
<!-- END MANUAL -->

## Tavily Map

### What it is
Maps a website's structure with Tavily, discovering its URLs without extracting content

### How it works
<!-- MANUAL: how_it_works -->
The block discovers the URLs reachable from a root URL, following links up to `max_depth` and returning a flat list of `links` — without fetching page content, which makes it faster and cheaper than a full crawl. Use `limit` to cap how many pages are discovered and optional natural-language `instructions` to narrow the results (for example, "Only documentation pages").

Because no content is extracted, mapping is billed at 1 credit per 10 discovered pages (2 per 10 when instructions are provided); actual spend is read from the API's usage report.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The root URL to map | str | Yes |
| limit | Maximum number of pages to discover | int | No |
| max_depth | Maximum link depth from the root URL | int | No |
| instructions | Natural language instructions guiding which pages to include (e.g. 'Only documentation pages') | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the map failed | str |
| links | List of URLs discovered on the website | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Sitemap Discovery**: Enumerate a site's pages before deciding what to crawl or extract.

**Scoped Crawl Planning**: Map first, filter the links, then feed a subset into the Extract block to control cost.

**Coverage Auditing**: List all reachable URLs under a section to spot orphaned or missing pages.
<!-- END MANUAL -->

---
