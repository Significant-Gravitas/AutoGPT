# Tavily Extract
<!-- MANUAL: file_description -->
Blocks for extracting clean page content from URLs with Tavily.
<!-- END MANUAL -->

## Tavily Extract

### What it is
Extracts page content from one or more URLs using Tavily, optimized for LLM consumption

### How it works
<!-- MANUAL: how_it_works -->
The block takes up to 20 URLs per request, fetches each page, and returns cleaned content in `markdown` or `text` format optimized for LLM consumption. Setting `extract_depth` to advanced retrieves more from each page, including tables and embedded content, at a higher credit cost.

Successfully extracted pages are returned on `results` (and one at a time on `result`), while any URLs that could not be fetched are surfaced separately on `failed_urls` — so a few bad URLs never fail the whole batch. Cost is 1 credit per 5 successfully extracted URLs (2 per 5 for advanced); failed extractions are free, and actual spend is read from the API's usage report.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| urls | The URLs to extract content from (up to 20 per request) | List[str] | Yes |
| extract_depth | Depth of the extraction: basic or advanced (retrieves more data, including tables and embedded content) | "basic" \| "advanced" | No |
| format | The format of the extracted content | "markdown" \| "text" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the extraction failed | str |
| results | List of successfully extracted pages | List[TavilyPageContent] |
| result | Single extracted page | TavilyPageContent |
| failed_urls | URLs that could not be extracted | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Content Ingestion**: Turn a list of article or documentation URLs into clean text for summarization or RAG indexing.

**Link Enrichment**: Follow up a search or map result by pulling the full content of the most relevant pages.

**Resilient Batch Scraping**: Extract many URLs at once and route any failures via `failed_urls` for retry without losing the successful pages.
<!-- END MANUAL -->

---
