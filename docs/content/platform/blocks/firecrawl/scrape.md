# Firecrawl Scrape

### What it is
Firecrawl scrapes a website to extract comprehensive data while bypassing blockers.

### What it does
Firecrawl scrapes a website to extract comprehensive data while bypassing blockers.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL to crawl | str | Yes |
| limit | The number of pages to crawl | int | No |
| only_main_content | Only return the main content of the page excluding headers, navs, footers, etc. | bool | No |
| max_age | The maximum age of the page in milliseconds - default is 1 hour | int | No |
| wait_for | Specify a delay in milliseconds before fetching the content, allowing the page sufficient time to load. | int | No |
| formats | The format of the crawl | List["markdown" | "html" | "rawHtml"] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the scrape failed | str |
| data | The result of the crawl | Dict[str, True] |
| markdown | The markdown of the crawl | str |
| html | The html of the crawl | str |
| raw_html | The raw html of the crawl | str |
| links | The links of the crawl | List[str] |
| screenshot | The screenshot of the crawl | str |
| screenshot_full_page | The screenshot full page of the crawl | str |
| json_data | The json data of the crawl | Dict[str, True] |
| change_tracking | The change tracking of the crawl | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
