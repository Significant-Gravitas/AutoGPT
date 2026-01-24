# Exa Contents
<!-- MANUAL: file_description -->
Blocks for retrieving and extracting content from web pages using Exa's contents API.
<!-- END MANUAL -->

## Exa Contents

### What it is
Retrieves document contents using Exa's contents API

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves full content from web pages using Exa's contents API. You can provide URLs directly or document IDs from previous searches. The API supports live crawling to fetch fresh content and can extract text, highlights, and AI-generated summaries.

The block supports subpage crawling to gather related content and offers various content retrieval options including full text extraction, relevant highlights, and customizable summary generation. Results are formatted for easy use with LLMs.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| urls | Array of URLs to crawl (preferred over 'ids') | List[str] | No |
| ids | [DEPRECATED - use 'urls' instead] Array of document IDs obtained from searches | List[str] | No |
| text | Retrieve text content from pages | bool | No |
| highlights | Text snippets most relevant from each page | HighlightSettings | No |
| summary | LLM-generated summary of the webpage | SummarySettings | No |
| livecrawl | Livecrawling options: never, fallback (default), always, preferred | "never" \| "fallback" \| "always" \| "preferred" | No |
| livecrawl_timeout | Timeout for livecrawling in milliseconds | int | No |
| subpages | Number of subpages to crawl | int | No |
| subpage_target | Keyword(s) to find specific subpages of search results | str \| List[str] | No |
| extras | Extra parameters for additional content | ExtrasSettings | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| results | List of document contents with metadata | List[ExaSearchResults] |
| result | Single document content result | ExaSearchResults |
| context | A formatted string of the results ready for LLMs | str |
| request_id | Unique identifier for the request | str |
| statuses | Status information for each requested URL | List[ContentStatus] |
| cost_dollars | Cost breakdown for the request | CostDollars |

### Possible use case
<!-- MANUAL: use_case -->
**Content Aggregation**: Retrieve full article content from multiple URLs for analysis or summarization.

**Competitive Research**: Crawl competitor websites to extract product information, pricing, or feature details.

**Data Enrichment**: Fetch detailed content from URLs discovered through Exa searches to build comprehensive datasets.
<!-- END MANUAL -->

---
