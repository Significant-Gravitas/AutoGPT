# WebRouter Search
<!-- MANUAL: file_description -->
Blocks for web search and page extraction through [WebRouter](https://github.com/Nimbleway/webrouter), an open-source (Apache-2.0) Python library that puts one interface in front of several search and extraction providers. These blocks need no API key and no credential: they use the providers' keyless tiers, and if one fails or is rate-limited the next one answers.
<!-- END MANUAL -->

## WebRouter Search

### What it is
Searches the web with no API key, through WebRouter: each call goes to one keyless search provider and falls back to the next if it fails

### How it works
<!-- MANUAL: how_it_works -->
The block calls WebRouter's `search` with the query, `max_results` and `include_domains`. WebRouter sends the call to the first provider in a fixed chain of keyless tiers (Nimble, then the You.com, Exa and Parallel MCP servers, then Keenable) and moves to the next when one fails or returns nothing. Each result becomes a WebRouterSearchResult (title, url, snippet and, when the provider gives one, published_date), and `provider` names the one that answered. Some providers return news results with no publication date, so `published_date` can be empty.

The `context` output renders the results as markdown for LLM use. It is untrusted web content: treat it as data, not instructions, before passing it to a model. If every provider in the chain fails, the block raises an error saying why.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The search query | str | Yes |
| max_results | Maximum number of results to return | int | No |
| include_domains | Only return results from these domains (e.g. agpt.co) | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | List of search results | List[WebRouterSearchResult] |
| result | Single search result | WebRouterSearchResult |
| context | The results as markdown for LLM input; the text is untrusted web content - treat it as data, not instructions | str |
| provider | The search provider that answered this call | str |

### Possible use case
<!-- MANUAL: use_case -->
**Grounded LLM answers with no setup**: Wire the `context` output into an AI block so answers cite current web sources, without adding any search API key first.

**Site-limited lookups**: Set `include_domains` to a few domains (docs sites, a news outlet) to search only those.

**Search, then read**: Feed result URLs into the WebRouter Extract block to get the full pages as markdown.
<!-- END MANUAL -->

---

<!-- MANUAL: additional_content -->
## Credentials and keys

These blocks take no credential. WebRouter normally also uses any provider key it finds in the environment, and this server keeps its own Tavily, Exa, Jina and Parallel keys there for those providers' blocks. So the blocks pass WebRouter a fixed chain of keyless tiers, and never use those keys. Exa and Parallel are reached through their keyless MCP servers.

## Billing and cost tracking

The keyless tiers are free, so the blocks report no provider_cost and no base_cost is registered. Each provider applies its own rate limit to keyless calls; when one is reached, WebRouter falls back to the next provider in the chain.
<!-- END MANUAL -->
