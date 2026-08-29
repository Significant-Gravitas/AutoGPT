# Firecrawl Search
<!-- MANUAL: file_description -->
Blocks for searching the web and extracting content using Firecrawl.
<!-- END MANUAL -->

## Firecrawl Search

### What it is
Firecrawl searches the web for the given query.

### How it works
<!-- MANUAL: how_it_works -->
This block uses Firecrawl's search API to find web pages matching your query and optionally extract their content. It performs a web search and can return results with full page content in your chosen format.

Configure the number of results to return, output formats (markdown, HTML, raw HTML), and caching behavior. The wait_for parameter allows time for JavaScript-heavy pages to fully render before extraction.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The query to search for | str | Yes |
| limit | The number of pages to crawl | int | No |
| max_age | The maximum age of the page in milliseconds - default is 1 hour | int | No |
| wait_for | Specify a delay in milliseconds before fetching the content, allowing the page sufficient time to load. | int | No |
| formats | Returns the content of the search if specified | List["markdown" \| "html" \| "rawHtml" \| "links" \| "screenshot" \| "screenshot@fullPage" \| "json" \| "changeTracking"] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the search failed | str |
| data | The result of the search | Dict[str, Any] |
| site | The site of the search | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Research Automation**: Search for topics and automatically extract content from relevant pages for analysis.

**Lead Generation**: Find companies or contacts matching specific criteria across the web.

**Content Aggregation**: Gather articles, reviews, or information on specific topics from multiple sources.
<!-- END MANUAL -->

---
