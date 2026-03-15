# Exa Similar
<!-- MANUAL: file_description -->
Blocks for finding semantically similar web pages using Exa's findSimilar API.
<!-- END MANUAL -->

## Exa Find Similar

### What it is
Finds similar links using Exa's findSimilar API

### How it works
<!-- MANUAL: how_it_works -->
This block uses Exa's findSimilar API to discover web pages that are semantically similar to a given URL. The API analyzes the content and context of the provided page to find related content across the web.

The block supports filtering by domains, date ranges, and text patterns to refine results. You can retrieve content directly with results and enable content moderation to filter unsafe content.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The url for which you would like to find similar links | str | Yes |
| number_of_results | Number of results to return | int | No |
| include_domains | List of domains to include in the search. If specified, results will only come from these domains. | List[str] | No |
| exclude_domains | Domains to exclude from search | List[str] | No |
| start_crawl_date | Start date for crawled content | str (date-time) | No |
| end_crawl_date | End date for crawled content | str (date-time) | No |
| start_published_date | Start date for published content | str (date-time) | No |
| end_published_date | End date for published content | str (date-time) | No |
| include_text | Text patterns to include (max 1 string, up to 5 words) | List[str] | No |
| exclude_text | Text patterns to exclude (max 1 string, up to 5 words) | List[str] | No |
| contents | Content retrieval settings | ContentSettings | No |
| moderation | Enable content moderation to filter unsafe content from search results | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| results | List of similar documents with metadata and content | List[ExaSearchResults] |
| result | Single similar document result | ExaSearchResults |
| context | A formatted string of the results ready for LLMs. | str |
| request_id | Unique identifier for the request | str |
| cost_dollars | Cost breakdown for the request | CostDollars |

### Possible use case
<!-- MANUAL: use_case -->
**Content Discovery**: Find related articles, blog posts, or resources similar to content you already like.

**Competitor Analysis**: Discover similar companies or products by finding pages similar to known competitors.

**Research Expansion**: Expand your research by finding additional sources similar to key reference materials.
<!-- END MANUAL -->

---
