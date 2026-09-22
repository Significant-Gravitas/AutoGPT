# Anysearch Parallel Search
<!-- MANUAL: file_description -->
Runs multiple AnySearch queries in a single block (client-side concurrency, max 5 queries). Requires an `ANYSEARCH_API_KEY` credential - see the AnySearch Search page for setup.
<!-- END MANUAL -->

## Any Search Parallel Search

### What it is
Runs several AnySearch queries in parallel (client-side concurrency via asyncio)

### How it works
<!-- MANUAL: how_it_works -->
The block fans out up to 5 queries client-side with asyncio.gather - each query is an independent POST to /v1/search, so this is client-side concurrency, not a server-side batch endpoint (the AnySearch REST surface does not expose one). Results return grouped per query in input order; a failing query does not fail the batch - the error field on that group carries the message while the other groups still return results. The domain, sub_domain and sub_domain_params inputs apply to every query in the batch.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| queries | The search queries to run in parallel (max 5) | List[str] | Yes |
| max_results | Maximum number of results per query | int | No |
| domain | Restrict every query to a vertical domain | "academic" \| "agriculture" \| "business" \| "code" \| "energy" \| "environment" \| "film" \| "finance" \| "gaming" \| "general" \| "health" \| "ip" \| "legal" \| "resource" \| "security" \| "social_media" \| "travel" | No |
| sub_domain | Sub-domain inside the domain, applied to every query (e.g. finance.quote) | str | No |
| sub_domain_params | Structured parameters for the shared sub_domain | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | One result group per query, in input order | List[AnySearchQueryResults] |

### Possible use case
<!-- MANUAL: use_case -->
**Multi-angle research**: Fan out 2-5 phrasings or sub-topics of one question in parallel, then merge the groups for an LLM synthesis step.

**Hybrid general + vertical**: Mix a general query with domain-scoped queries when you are not sure which surface answers best.

**Per-sub-domain coverage**: After discovering sub-domains, probe several of them in one step - e.g. finance.news and finance.quote for the same ticker.
<!-- END MANUAL -->

---
