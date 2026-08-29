# Firecrawl Scrape
<!-- MANUAL: file_description -->
Blocks for scraping individual web pages and extracting content using Firecrawl.
<!-- END MANUAL -->

## Firecrawl Scrape

### What it is
Firecrawl scrapes a website to extract comprehensive data while bypassing blockers.

### How it works
<!-- MANUAL: how_it_works -->
This block uses Firecrawl's scraping API to extract content from a single URL. It handles JavaScript rendering, bypasses anti-bot measures, and can return content in multiple formats including markdown, HTML, and screenshots.

Configure output formats, filter to main content only, and set wait times for dynamic pages. The block returns comprehensive results including extracted content, links found on the page, and optional change tracking data.
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
| error | Error message if the scrape failed | str |
| data | The result of the crawl | Dict[str, Any] |
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
**Article Extraction**: Scrape news articles or blog posts to extract clean, readable content.

**Price Monitoring**: Regularly scrape product pages to track price changes over time.

**Content Backup**: Create markdown backups of important web pages for offline reference.
<!-- END MANUAL -->

---
