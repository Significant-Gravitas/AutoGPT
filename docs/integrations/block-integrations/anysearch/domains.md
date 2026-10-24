# AnySearch Domains
<!-- MANUAL: file_description -->
Discovers the live AnySearch domain and sub-domain catalog - names, descriptions, and the parameter contract of every sub-domain - via the vendor's public GET endpoints. Runs anonymously or with an `ANYSEARCH_API_KEY` credential - see the AnySearch Search page for setup.
<!-- END MANUAL -->

## AnySearch Domains

### What it is
Discovers the AnySearch domain and sub-domain catalog (sub-domain names and their parameter contracts) via the public GET catalog endpoints

### How it works
<!-- MANUAL: how_it_works -->
With `domains` left empty the block calls `GET https://api.anysearch.com/v1/domains` and returns every domain with its description and `sub_domain_count`. With one or more `domains` selected it instead calls `GET https://api.anysearch.com/v1/sub-domains?domain=a&domain=b` - the `domain` query parameter is repeated per value, never comma-joined, as the endpoint requires - and returns the selected domains with their `sub_domains` arrays, where each sub-domain carries its `params` contract (name, description, required flag).

Both endpoints are public catalog reads: they need no API key and consume no quota, so this block is safe to run before every search. The selectable domain list mirrors a snapshot of the upstream catalog and can drift: if AnySearch retires a domain the list still offers, selecting it fails the whole call with an upstream 5xx error - one stale name poisons the batch rather than returning partial results - and the block's error carries a stale-catalog hint. An empty `sub_domains` array on a returned domain simply means that domain currently exposes no sub-domains. The `topology_version` output pins the catalog snapshot the answer came from; sub-domains and their parameters evolve, so record the version when persisting discovered values. The `request_id` output pins the server-side request identifier echoed in the response envelope.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| auth | Anonymous tier (lower rate limit, no key) or an AnySearch API key credential | "api_key" \| "anonymous" | No |
| domains | Restrict discovery to these domains; leave empty to list the full catalog | List["academic" \| "agriculture" \| "business" \| "code" \| "energy" \| "environment" \| "film" \| "finance" \| "gaming" \| "general" \| "health" \| "ip" \| "legal" \| "resource" \| "security" \| "social_media" \| "travel"] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| domains | Catalog entries as returned by the API - every domain for an unrestricted call, or the selected domains with their sub-domains and parameter contracts | List[Dict[str, Any]] |
| topology_version | Catalog snapshot version; sub-domains evolve, so note it when recording discovered values | int |
| request_id | Unique identifier for this request | str |

### Possible use case
<!-- MANUAL: use_case -->
**Runtime sub-domain discovery**: Run this block before a search block in the same graph so valid `sub_domain` tags and required `sub_domain_params` are resolved from the live catalog instead of a stale hardcoded list.

**Parameter-contract inspection**: Select one or two domains (e.g. `finance`, `code`) to inspect each sub-domain's `params` - which keys are required, what values they take - before wiring a vertical query.

**Catalog health check**: Diff the `domains` list and `topology_version` between runs to detect when AnySearch added or renamed a capability a graph depends on.
<!-- END MANUAL -->

---
