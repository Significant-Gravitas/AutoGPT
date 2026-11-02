# Tavily Crawl
<!-- MANUAL: file_description -->
Blocks for crawling a website and extracting page content with Tavily.
<!-- END MANUAL -->

## Tavily Crawl

### What it is
Crawls a website with Tavily, following links from the root URL and extracting page content

### How it works
<!-- MANUAL: how_it_works -->
Starting from a root URL, the block follows links up to `max_depth` and extracts the content of each page it visits, stopping at `limit` pages. Optional natural-language `instructions` steer which pages to follow (for example, "Find all the API reference pages"), and `extract_depth` and `format` control how much content is captured and whether it comes back as markdown or text.

Each crawled page is returned on `results` (and one at a time on `result`) with its extracted content. A crawl combines mapping and extraction, so its credit cost is the sum of both per Tavily's schedule; actual spend is read from the API's usage report.
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
**Documentation Ingestion**: Crawl a docs site and extract every page to build a knowledge base or RAG pipeline.

**Targeted Site Harvesting**: Use `instructions` to gather only pricing, product, or reference pages from a large site.

**Competitive Content Snapshots**: Capture the content of a section of a site in one pass for analysis or change tracking.
<!-- END MANUAL -->

---
