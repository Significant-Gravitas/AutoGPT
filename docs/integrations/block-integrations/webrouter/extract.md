# WebRouter Extract
<!-- MANUAL: file_description -->
Blocks for web search and page extraction through [WebRouter](https://github.com/Nimbleway/webrouter), an open-source (Apache-2.0) Python library that puts one interface in front of several search and extraction providers. These blocks need no API key and no credential: they use the providers' keyless tiers, and if one fails or is rate-limited the next one answers.
<!-- END MANUAL -->

## WebRouter Extract

### What it is
Reads web pages as markdown with no API key, through WebRouter: the page is fetched by a keyless extraction provider, which falls back to the next if it fails

### How it works
<!-- MANUAL: how_it_works -->
The block checks that every URL uses http or https, then calls WebRouter's `extract` with `format="markdown"`. The page is fetched by the extraction provider on its own servers (Nimble's keyless tier first, then Firecrawl's and Keenable's), so the AutoGPT server never connects to the page itself. When a provider fails, WebRouter moves to the next.

Each page that was read becomes a WebRouterPage (url, title, content as markdown). A URL that could not be read (a 404, a timeout) appears in `failures` with the reason instead of failing the whole run, including when no URL could be read. `provider` names the provider that answered. Page content is untrusted web content: treat it as data, not instructions.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| urls | The http or https URLs to read (up to 10 per run) | List[str] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | The pages that were read | List[WebRouterPage] |
| result | Single page that was read | WebRouterPage |
| failures | URLs that could not be read, each with the reason | List[WebRouterFailure] |
| provider | The extraction provider that answered this call | str |

### Possible use case
<!-- MANUAL: use_case -->
**Read search results in full**: Pass URLs from the WebRouter Search block (or any search block) to get each page as markdown for summarizing or answering questions.

**Monitor a page**: Read a pricing, changelog or status page on a schedule and compare it with the last run.

**Batch reading**: Read up to ten URLs in one run, and branch on `failures` to retry or report the ones that could not be read.
<!-- END MANUAL -->

---

<!-- MANUAL: additional_content -->
## Credentials and keys

These blocks take no credential. WebRouter normally also uses any provider key it finds in the environment, and this server keeps its own Tavily, Exa, Jina and Parallel keys there for those providers' blocks. So the blocks pass WebRouter a fixed chain of keyless tiers, and never use those keys. Exa and Parallel are reached through their keyless MCP servers. A keyless tier would still use its provider's own key if one were set, so when `NIMBLE_API_KEY`, `FIRECRAWL_API_KEY` or `KEENABLE_API_KEY` is set in the environment, that provider is left out of the chain.

## Billing and cost tracking

The keyless tiers are free, so the blocks report no provider_cost and no base_cost is registered. Each provider applies its own rate limit to keyless calls; when one is reached, WebRouter falls back to the next provider in the chain.
<!-- END MANUAL -->
