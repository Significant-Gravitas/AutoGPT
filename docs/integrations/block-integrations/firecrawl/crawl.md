# Firecrawl Crawl
<!-- MANUAL: file_description -->
Blocks for crawling multiple pages of a website using Firecrawl.
<!-- END MANUAL -->

## Firecrawl Crawl

### What it is
Firecrawl crawls websites to extract comprehensive data while bypassing blockers.

### How it works
<!-- MANUAL: how_it_works -->
This block uses Firecrawl's API to crawl multiple pages of a website starting from a given URL. It navigates through links, handling JavaScript rendering and bypassing anti-bot measures to extract clean content from each page.

Configure the crawl depth with the limit parameter, choose output formats (markdown, HTML, or raw HTML), and optionally filter to main content only. The block supports caching with configurable max age and wait times for dynamic content.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL to crawl | str | Yes |
| limit | The number of pages to crawl | int | No |
| only_main_content | Only return the main content of the page excluding headers, navs, footers, etc. | bool | No |
| max_age | The maximum age of the page in milliseconds - default is 1 hour | int | No |
| wait_for | Specify a delay in milliseconds before fetching the content, allowing the page sufficient time to load. | int | No |
| formats | The format of the crawl | List["markdown" \| "html" \| "rawHtml" \| "links" \| "screenshot" \| "screenshot@fullPage" \| "json" \| "changeTracking"] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the crawl failed | str |
| data | The result of the crawl | List[Dict[str, Any]] |
| markdown | The markdown of the crawl | str |
| html | The html of the crawl | str |
| raw_html | The raw html of the crawl | str |
| links | The links of the crawl | List[str] |
| screenshot | The screenshot of the crawl | str |
| screenshot_full_page | The screenshot full page of the crawl | str |
| json_data | The json data of the crawl | Dict[str, Any] |
| change_tracking | The change tracking of the crawl | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Documentation Indexing**: Crawl entire documentation sites to build searchable knowledge bases or training data.

**Competitor Research**: Extract content from competitor websites for market analysis and comparison.

**Content Archival**: Systematically archive website content for backup or compliance purposes.
<!-- END MANUAL -->

---
