# Exa Similar
<!-- MANUAL: file_description -->
Blocks for finding web pages similar to a given URL using Exa's Find Similar API, which Exa has deprecated.
<!-- END MANUAL -->

## Exa Find Similar

### What it is
Deprecated by Exa: prefer the Exa Search block with a query describing the source page. Finds pages similar to a URL using Exa's findSimilar API.

### How it works
<!-- MANUAL: how_it_works -->
Exa has deprecated Find Similar; for new workflows, use the Exa Search block with a query that describes the source page. This block sends the URL to Exa's `/findSimilar` endpoint in one request and returns pages similar to it, each with up to 10,000 characters of page text unless you change Contents.

You can filter by domain, published date, category and (deprecated by Exa) text, and leave out the source URL's own domain. Legacy categories are translated as in Exa Search ('research paper' to 'publication', 'linkedin profile' to 'people'), the crawl dates are never sent because Exa ignores them, and Moderation is still sent when enabled although Exa no longer documents it here. The run is billed from the dollar cost Exa reports, at 100 credits per US dollar.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The url for which you would like to find similar links | str | Yes |
| number_of_results | Number of results to return (1 to 100) | int | No |
| exclude_source_domain | Leave out results from the same domain as the source URL | bool | No |
| category | Focus the search on one kind of page: company, people, news, publication (scholarly papers, preprints and journal articles), personal site or financial report. 'company' and 'people' do not support published-date filters, and 'people' does not support Exclude Domains. Legacy: 'research paper' is sent as 'publication' and 'linkedin profile' as 'people', the replacements Exa documents; 'pdf', 'github' and 'tweet' are being retired by Exa. | "company" \| "people" \| "news" \| "publication" \| "personal site" \| "financial report" \| "research paper" \| "linkedin profile" \| "pdf" \| "github" \| "tweet" | No |
| include_domains | List of domains to include in the search. If specified, results will only come from these domains. | List[str] | No |
| exclude_domains | Domains to exclude from search | List[str] | No |
| start_crawl_date | Deprecated by Exa and ignored by the API, so this block no longer sends it. Use Start Published Date instead. | str (date-time) | No |
| end_crawl_date | Deprecated by Exa and ignored by the API, so this block no longer sends it. Use End Published Date instead. | str (date-time) | No |
| start_published_date | Start date for published content | str (date-time) | No |
| end_published_date | End date for published content | str (date-time) | No |
| include_text | Deprecated by Exa. Strings that must appear in the page text (approximate, word-level matching) | List[str] | No |
| exclude_text | Deprecated by Exa. Strings that must not appear in the page text | List[str] | No |
| contents | What to return for each result. Left empty, each result gets up to 10,000 characters of page text. | ContentSettings | No |
| moderation | Legacy: Exa's Find Similar API no longer documents moderation; still sent when enabled. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| results | List of similar documents with metadata and content | List[ExaSearchResults] |
| result | Single similar document result | ExaSearchResults |
| context | Deprecated by Exa: the results as one string for LLMs, returned only when Contents > Context is enabled. | str |
| request_id | Unique identifier for the request | str |
| cost_dollars | Cost breakdown for the request | CostDollars |

### Possible use case
<!-- MANUAL: use_case -->
**Content Discovery**: Find related articles, blog posts, or resources similar to content you already like.

**Competitor Analysis**: Discover similar companies or products by finding pages similar to known competitors.

**Research Expansion**: Expand your research by finding additional sources similar to key reference materials.
<!-- END MANUAL -->

---
