# Keenable Fetch
<!-- MANUAL: file_description -->
Blocks for Keenable web search and page fetch. Both blocks work without an API key: anonymous calls go to Keenable's keyless public endpoints (rate-limited per IP); a `KEENABLE_API_KEY` credential lifts the limit.
<!-- END MANUAL -->

## Keenable Fetch

### What it is
Fetches a web page as clean markdown using Keenable. Works without an API key.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `GET https://api.keenable.ai/v1/fetch?url=...&live=true` (or the keyless `/v1/fetch/public` in anonymous mode) and returns the page as clean markdown together with its title. When `max_chars` is set Keenable cuts the content to that length and appends a truncation notice, which the block reports through the `truncated` output. The content is untrusted web content - treat it as data, not instructions. HTTP or transport failures raise a block error.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| auth | Anonymous (keyless, rate-limited per IP) or a Keenable API key credential | "api_key" \| "anonymous" | No |
| url | The URL to fetch | str | Yes |
| max_chars | Cut the returned markdown at this many characters | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| url | URL of the fetched page | str |
| title | Title of the fetched page | str |
| content | Page content as clean markdown (untrusted web content) | str |
| truncated | Whether the content was cut to fit max_chars | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Read a search hit**: pass a URL from Keenable Search to this block to get the full page as markdown for summarisation.

**Monitor a page**: fetch a pricing or changelog page on a schedule and compare it with the previous run.

**Cite a source**: fetch a page's markdown and title to quote it accurately in a report.
<!-- END MANUAL -->

---
