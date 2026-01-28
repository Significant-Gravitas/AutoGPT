# Exa Search
<!-- MANUAL: file_description -->
Blocks for searching the web using Exa's advanced neural and keyword search API.
<!-- END MANUAL -->

## Exa Search

### What it is
Searches the web using Exa's advanced search API

### How it works
<!-- MANUAL: how_it_works -->
This block uses Exa's advanced search API to find web content. Unlike traditional search engines, Exa offers neural search that understands semantic meaning, making it excellent for finding specific types of content. You can choose between keyword search (traditional), neural search (semantic understanding), or fast search.

The block supports powerful filtering by domain, date ranges, content categories (companies, research papers, news, etc.), and text patterns. Results include URLs, titles, and optionally full content extraction.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The search query | str | Yes |
| type | Type of search | "keyword" \| "neural" \| "fast" \| "auto" | No |
| category | Category to search within: company, research paper, news, pdf, github, tweet, personal site, linkedin profile, financial report | "company" \| "research paper" \| "news" \| "pdf" \| "github" \| "tweet" \| "personal site" \| "linkedin profile" \| "financial report" | No |
| user_location | The two-letter ISO country code of the user (e.g., 'US') | str | No |
| number_of_results | Number of results to return | int | No |
| include_domains | Domains to include in search | List[str] | No |
| exclude_domains | Domains to exclude from search | List[str] | No |
| start_crawl_date | Start date for crawled content | str (date-time) | No |
| end_crawl_date | End date for crawled content | str (date-time) | No |
| start_published_date | Start date for published content | str (date-time) | No |
| end_published_date | End date for published content | str (date-time) | No |
| include_text | Text patterns to include | List[str] | No |
| exclude_text | Text patterns to exclude | List[str] | No |
| contents | Content retrieval settings | ContentSettings | No |
| moderation | Enable content moderation to filter unsafe content from search results | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| results | List of search results | List[ExaSearchResults] |
| result | Single search result | ExaSearchResults |
| context | A formatted string of the search results ready for LLMs. | str |
| search_type | For auto searches, indicates which search type was selected. | str |
| resolved_search_type | The search type that was actually used for this request (neural or keyword) | str |
| cost_dollars | Cost breakdown for the request | CostDollars |

### Possible use case
<!-- MANUAL: use_case -->
**Competitive Research**: Search for companies in a specific industry, filtered by recent news or funding announcements.

**Content Curation**: Find relevant articles and research papers on specific topics for newsletters or content aggregation.

**Lead Generation**: Search for companies matching specific criteria (industry, size, recent activity) for sales prospecting.
<!-- END MANUAL -->

---
