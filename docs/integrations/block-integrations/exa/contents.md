# Exa Contents
<!-- MANUAL: file_description -->
Blocks for retrieving and extracting content from web pages using Exa's contents API.
<!-- END MANUAL -->

## Exa Contents

### What it is
Fetches clean text, highlights or summaries for known URLs with Exa Contents, using cached copies or fresh fetches

### How it works
<!-- MANUAL: how_it_works -->
This block sends the URLs (or, if no URLs are given, the deprecated document IDs) to Exa's `/contents` endpoint in one request and returns each page's text, plus highlights, a summary, subpages or extra links when you ask for them. It fails before calling Exa if neither URLs nor IDs are set. Statuses reports per URL whether the fetch succeeded and whether the page came from Exa's cache or a fresh crawl.

Max Age Hours replaces the deprecated Livecrawl option and wins when both are set. Legacy Livecrawl values are translated: 'always' is sent as max age 0, 'never' as -1, 'fallback' (the default) as Exa's default, and 'preferred' is passed through unchanged. Snapshot As Of returns stored page versions only and is rejected together with Max Age Hours or a translated Livecrawl value; the run is billed from the dollar cost Exa reports, at 100 credits per US dollar.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| urls | Up to 100 page URLs to fetch (preferred over 'ids') | List[str] | No |
| ids | [DEPRECATED - use 'urls' instead] Array of document IDs obtained from searches | List[str] | No |
| text | Return each page's text | bool | No |
| text_max_characters | Maximum characters of text per page (1 to 1,000,000). Leave empty for the full page. | int | No |
| text_include_html_tags | Keep lightweight HTML tags in the text so an LLM can see the page structure | bool | No |
| text_verbosity | How much of each page to render: 'compact' (Exa's default) keeps the main content, 'standard' adds surrounding context, 'full' returns the most complete text | "compact" \| "standard" \| "full" | No |
| text_include_sections | Only keep text from these page sections (best effort) | List["header" \| "navigation" \| "banner" \| "body" \| "sidebar" \| "footer" \| "metadata"] | No |
| text_exclude_sections | Drop text from these page sections, e.g. navigation and footer (best effort) | List["header" \| "navigation" \| "banner" \| "body" \| "sidebar" \| "footer" \| "metadata"] | No |
| highlights | The passages of each page most relevant to a query. Sent only when a query, max characters or (deprecated) sentence count is set | HighlightSettings | No |
| summary | LLM-generated summary of each page (billed per page). Sent only when a query or schema is set | SummarySettings | No |
| max_age_hours | Maximum age of cached content, in hours (-1 to 720). Older pages are fetched fresh; 0 always fetches fresh, -1 only uses the cache. Leave empty (recommended) to use the cache and fetch pages that have no cached copy. Overrides Livecrawl. | int | No |
| snapshot_as_of | Return each page as Exa stored it at this time: an ISO 8601 date-time or a date (midnight UTC). Never fetches live pages, so it cannot be combined with Max Age Hours, Livecrawl or Subpages. | str | No |
| livecrawl | Deprecated by Exa: use Max Age Hours instead. Translated as always -> 0, never -> -1, fallback (default) -> Exa's default; 'preferred' has no equivalent and is passed through. | "never" \| "fallback" \| "always" \| "preferred" | No |
| livecrawl_timeout | How long a fresh page fetch may take, in milliseconds (1 to 90000) | int | No |
| subpages | Number of subpages to crawl per page (0 to 100) | int | No |
| subpage_target | Keyword(s) to find specific subpages of search results | str \| List[str] | No |
| extras | Extra links and images to return for each page | ExtrasSettings | No |
| user_identifier | Opaque ID of the end user this request is for (1-64 letters, digits, '_' or '-', e.g. a UUID or hash). Never an email or name. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| results | List of document contents with metadata | List[ExaSearchResults] |
| result | Single document content result | ExaSearchResults |
| context | Deprecated by Exa: the pages as one string for LLMs. Prefer the text or highlights of each result. | str |
| request_id | Unique identifier for the request | str |
| statuses | Status information for each requested URL | List[ContentStatus] |
| cost_dollars | Cost breakdown for the request | CostDollars |

### Possible use case
<!-- MANUAL: use_case -->
**Content Aggregation**: Retrieve full article content from multiple URLs for analysis or summarization.

**Competitive Research**: Fetch competitor pages fresh with Max Age Hours set to 0 to extract current product and pricing details.

**Data Enrichment**: Fetch detailed content from URLs discovered through Exa searches to build comprehensive datasets.
<!-- END MANUAL -->

---
