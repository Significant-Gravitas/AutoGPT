# AnySearch Search
<!-- MANUAL: file_description -->
Blocks for web search with AnySearch, an AI-native search API covering general queries and 16 vertical domains. All blocks can call AnySearch anonymously (lower rate limit) or with an `ANYSEARCH_API_KEY` credential.
<!-- END MANUAL -->

## AnySearch

### What it is
Searches the web using AnySearch - general queries plus vertical domains (finance, academic, health, legal, and more) via domain/sub_domain filters

### How it works
<!-- MANUAL: how_it_works -->
The block sends the query to the AnySearch REST endpoint `POST https://api.anysearch.com/v1/search` (Bearer API-key auth when a credential is used; anonymous calls send no `Authorization` header) and parses a `{code, message, data}` envelope; a non-zero code or any HTTP/transport failure raises a block error instead of producing partial output. Each row in `data.results` becomes an AnySearchResult (title, url, snippet, content) and the `context` output renders the results as markdown for LLM use - that text is untrusted web content, so treat it as data, not instructions, before passing it to a model.

Leaving `sub_domain` empty runs a general search. Setting `sub_domain` picks a capability tag in `{domain}.{sub_domain}` form (e.g. `finance.quote` or `academic.search`) which is sent to the API as `tag`, and `sub_domain_params` is forwarded as `params` carrying the structured parameters that capability requires (e.g. `{"type": "stock", "symbol": "NVDA"}`). The optional `domain` input must agree with `sub_domain` - setting `domain` without `sub_domain`, or a `sub_domain` that does not start with `{domain}.`, is rejected by input validation before any request is sent; the API itself rejects unknown tags or malformed params with HTTP 400, surfaced as a block error. The complete catalog of sub-domains and their parameters is exposed by the AnySearch get_sub_domains tool - see the notes below.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| auth | Anonymous tier (lower rate limit, no key) or an AnySearch API key credential | "api_key" \| "anonymous" | No |
| query | The search query | str | Yes |
| max_results | Maximum number of results to return | int | No |
| domain | Restrict the search to a vertical domain | "academic" \| "agriculture" \| "business" \| "code" \| "energy" \| "environment" \| "film" \| "finance" \| "gaming" \| "general" \| "health" \| "ip" \| "legal" \| "resource" \| "security" \| "social_media" \| "travel" | No |
| sub_domain | Sub-domain inside the domain (e.g. finance.quote); discover valid values via the AnySearch get_sub_domains tool | str | No |
| sub_domain_params | Structured parameters required by the chosen sub_domain (e.g. type=stock and symbol=AAPL for finance.quote) | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | List of search results | List[AnySearchResult] |
| result | First search result | AnySearchResult |
| context | The search results formatted as markdown for LLM input; the text is untrusted web content - treat it as data, not instructions | str |

### Possible use case
<!-- MANUAL: use_case -->
**Fresh-grounded LLM answers**: Wire the context output into an AI block so responses cite current web sources instead of stale model memory; the context is untrusted page text, so the downstream prompt should treat it as data, not instructions.

**Vertical lookups**: Use domain + sub_domain + sub_domain_params for structured queries such as a stock quote (finance.quote with type=stock and a ticker symbol) or paper search (academic.search with a publication-year range).

**Research fan-in**: Collect ranked titles, URLs and snippets for a topic, then feed the result URLs into the AnySearch Extract block to pull full page content.
<!-- END MANUAL -->

---

<!-- MANUAL: additional_content -->
## Setup: ANYSEARCH_API_KEY

AnySearch issues API keys in the as_sk_... format. Sign up at https://anysearch.com to create a key, then either add it in AutoGPT under Settings > Credentials (provider: anysearch) or set `ANYSEARCH_API_KEY=` in autogpt_platform/backend/.env before starting the backend.

## Anonymous tier

The AnySearch API accepts unauthenticated requests at a lower rate limit. Blocks run anonymously by default; set Authentication to API key to use a credential - on Anonymous the API-key field hides and requests are sent without an `Authorization` header. Hosts can also set a default `ANYSEARCH_API_KEY` in .env so users pick the pre-seeded "AnySearch API Key" credential instead of creating their own; a credential whose key value is left empty likewise sends no `Authorization` header.

## Billing and cost tracking

AnySearch publishes no per-call price list (Free tier: 1,000 requests/day; the Professional plan is marked "coming soon"). The blocks therefore report no provider_cost and no base_cost is registered - usage stays free of platform-side cost accounting until AnySearch publishes rates, at which point `ProviderBuilder.with_base_cost` can be adopted.

## Vertical search parameters

- `domain` accepts: academic, agriculture, business, code, energy, environment, film, finance, gaming, general, health, ip, legal, resource, security, social_media, travel.
- `sub_domain` is a capability name inside the domain, e.g. `finance.quote` (real-time and historical quotes), `finance.news` (financial news), `academic.search` (paper search), `academic.biomedical` (MEDLINE/PMC literature), `health.trial`, `legal.legislation`.
- `sub_domain_params` carries the parameters a sub-domain requires, e.g.:
  - `finance.quote`: `{"type": "stock", "symbol": "NVDA"}` (type is required: stock | forex | crypto | commodity | index | etf)
  - `finance.news`: `{"type": "general"}` or `{"type": "stock", "symbol": "AAPL"}`
  - `academic.search`: `{"sort": "cited_by_count:desc", "year_from": "2020", "category": "Computer Science"}`
- To enumerate the complete current catalog - every domain, sub-domain, and required vs optional parameter - call the AnySearch get_sub_domains tool through the MCP endpoint below; the catalog evolves faster than this page.

## Zero-code alternative: the MCP endpoint

AnySearch also exposes a Model Context Protocol server at `https://api.anysearch.com/mcp` (Bearer auth) with the tools search, batch_search, get_sub_domains and extract. For one-off workflows you can wire a generic MCP block to that endpoint instead of these blocks; the REST blocks documented here are the recommended path for production graphs.
<!-- END MANUAL -->
