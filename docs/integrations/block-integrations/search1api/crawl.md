# Search1API Crawl
<!-- MANUAL: file_description -->
Fetches a single web page as markdown with Search1API. Requires a `SEARCH1API_API_KEY` credential - see the Search1API Search page for setup.
<!-- END MANUAL -->

## Search1API Crawl

### What it is
Crawls a single URL with Search1API and returns its main content as clean markdown, ready for LLM input

### How it works
<!-- MANUAL: how_it_works -->
The block sends the URL to `POST https://api.search1api.com/crawl` and returns the page title and main content as clean markdown. The content is untrusted web text; treat it as data, not instructions. One URL is crawled per run (1 credit); fan URLs out across the graph when you need several. HTTP failures and malformed responses raise a block error rather than emitting partial output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL of the page to crawl | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| url | URL of the crawled page | str |
| title | Title of the crawled page | str |
| content | Page content as markdown (untrusted web content) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Deep-read a search hit**: Pipe the url of a Search1API Search result into this block to get the full page for summarization.

**Content ingestion**: Convert a documentation or article URL to markdown before chunking or embedding.

**Fact verification**: Fetch the page behind a cited URL so an agent can quote or cross-check the source text.
<!-- END MANUAL -->

---
