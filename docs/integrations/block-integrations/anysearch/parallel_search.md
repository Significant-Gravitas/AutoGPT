# AnySearch Parallel Search
<!-- MANUAL: file_description -->
Runs multiple AnySearch queries in a single block (client-side concurrency, max 5 queries). Runs anonymously or with an `ANYSEARCH_API_KEY` credential - see the AnySearch Search page for setup.
<!-- END MANUAL -->

## AnySearch Parallel Search

### What it is
Runs several AnySearch queries in parallel (client-side concurrency via asyncio)

### How it works
<!-- MANUAL: how_it_works -->
The block fans out up to 5 queries client-side with asyncio.gather - each query is an independent POST to /v1/search, so this is client-side concurrency, not a server-side batch endpoint (the AnySearch REST surface does not expose one). Results return grouped per query in input order. Expected failures - envelope errors (a non-zero `code`), HTTP status failures, transport errors and timeouts - do not fail the batch: the error field on that query's group carries the message while the other groups still return results, and a group with an empty error succeeded. An unexpected exception instead fails the whole block, cancelling the remaining queries and waiting for their cleanup before the error propagates. Each group also carries its own `request_id` - the server-side identifier of that query's HTTP call - when AnySearch supplies one. HTTP failures retain their status and a validated UUID request ID from the response envelope or `X-Request-ID` header; upstream error details are withheld because they can contain credentials. A query with no valid response ID (for example a transport error) reports an empty `request_id`. A successful response missing `data.results` degrades to an error group rather than an empty result set.

The shared inputs apply to every query in the batch: `sub_domain` (e.g. `finance.quote`) is sent as `tag`, `sub_domain_params` as `params`, and the optional `zone`, `language` and `format` inputs are forwarded to all five calls (see the AnySearch Search page for their semantics). Setting `domain` without `sub_domain`, or a `sub_domain` outside the domain prefix, is rejected by input validation before any request is sent. Providing more than five queries, or an empty list, is likewise rejected by input validation (`queries` allows 1-5 entries).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| auth | Anonymous tier (lower rate limit, no key) or an AnySearch API key credential | "api_key" \| "anonymous" | No |
| queries | The search queries to run in parallel (max 5) | List[str] | Yes |
| max_results | Maximum number of results per query | int | No |
| domain | Restrict every query to a vertical domain | "academic" \| "agriculture" \| "business" \| "code" \| "energy" \| "environment" \| "film" \| "finance" \| "gaming" \| "general" \| "health" \| "ip" \| "legal" \| "resource" \| "security" \| "social_media" \| "travel" | No |
| sub_domain | Sub-domain inside the domain, applied to every query (e.g. finance.quote) | str | No |
| sub_domain_params | Structured parameters for the shared sub_domain | Dict[str, Any] | No |
| zone | Regional recall-pool routing (cn=China, intl=international) applied to every query; the effect varies by query | "cn" \| "intl" | No |
| language | Preferred result language as a BCP 47 tag (e.g. zh-CN, en), applied to every query | str | No |
| format | Content encoding inside each result - 'markdown' makes AnySearchResult.content Markdown text; the outer response stays JSON | "json" \| "markdown" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | One result group per query, in input order | List[AnySearchQueryResults] |

### Possible use case
<!-- MANUAL: use_case -->
**Multi-angle research**: Fan out 2-5 phrasings or sub-topics of one question in parallel, then merge the groups for an LLM synthesis step.

**Single-vertical sweep**: Run several phrasings inside one shared vertical - e.g. domain=finance with sub_domain=finance.quote and sub_domain_params {"type": "stock", "symbol": "NVDA"} applied to earnings, guidance, and analyst-rating queries at once.

**Bounded batch retrieval**: Fetch up to 5 independent lookups in one node while sharing the same credential and result shape.
<!-- END MANUAL -->

---
