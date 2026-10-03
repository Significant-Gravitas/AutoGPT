# Linkup Fetch
<!-- MANUAL: file_description -->
Blocks for fetching web pages as clean markdown with Linkup.
<!-- END MANUAL -->

## Linkup Fetch

### What it is
Fetches a web page using Linkup and returns its content as markdown

### How it works
<!-- MANUAL: how_it_works -->
The block sends the URL to Linkup's fetch endpoint and returns the page content as clean markdown, stripped of navigation and boilerplate so it can be passed straight to an LLM. Enable `render_js` for client-rendered pages that only show their content after JavaScript runs; this is slower and costs more per call. The documented per-call price is reported to the platform's cost tracking.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL of the web page to fetch | str | Yes |
| render_js | Render the page's JavaScript before extracting content (slower, needed for client-rendered pages) | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the fetch failed | str |
| markdown | The page content as clean markdown, ready for LLMs | str |

### Possible use case
<!-- MANUAL: use_case -->
**Read a Search Result**: Pass a `result` URL from the Linkup Search block to get the full page content for deeper analysis.

**Content Ingestion**: Convert documentation, articles or product pages to markdown before chunking and embedding them into a knowledge base.

**Page Monitoring**: Fetch a page on a schedule and compare its markdown over time to detect changes.
<!-- END MANUAL -->

---
