# Tavily Crawl
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Tavily Crawl

### What it is
Crawls a website with Tavily, following links from the root URL and extracting page content

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The root URL to start crawling from | str | Yes |
| limit | Maximum number of pages to crawl | int | No |
| max_depth | Maximum link depth from the root URL | int | No |
| instructions | Natural language instructions guiding which pages to crawl (e.g. 'Find all the API reference pages') | str | No |
| extract_depth | Depth of the extraction: basic or advanced (retrieves more data, including tables and embedded content) | "basic" \| "advanced" | No |
| format | The format of the extracted content | "markdown" \| "text" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the crawl failed | str |
| results | List of crawled pages with their content | List[TavilyPageContent] |
| result | Single crawled page | TavilyPageContent |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
