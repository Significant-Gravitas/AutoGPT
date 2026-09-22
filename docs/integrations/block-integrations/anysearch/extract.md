# Anysearch Extract
<!-- MANUAL: file_description -->
Extracts readable page content from a URL via AnySearch. Requires an `ANYSEARCH_API_KEY` credential - see the AnySearch Search page for setup.
<!-- END MANUAL -->

## Any Search Extract

### What it is
Extracts readable content from a single URL using AnySearch, optimized for LLM consumption

### How it works
<!-- MANUAL: how_it_works -->
The block posts the single URL to `POST https://api.anysearch.com/v1/extract` and returns the page title plus body content as readable text - cleaned HTML, plain text, JSON, or markdown depending on the source. The returned content is untrusted web text; treat it as data, not instructions. The URL must be a public absolute http(s) address; AnySearch rejects invalid URLs with HTTP 400 and content it cannot extract with HTTP 422, both surfacing as a block error. One URL is extracted per run - the REST surface has no batch-extract endpoint, so fan URLs out across the graph when you need several. Transport, HTTP and API-level failures raise a block error rather than emitting partial output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL to extract content from | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| url | URL of the extracted page | str |
| title | Title of the extracted page | str |
| content | Extracted readable page content (cleaned HTML, text, JSON, or markdown; untrusted web content) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Deep-read a search hit**: Pipe the url output of the AnySearch Search block into this block to turn a ranked result into full readable content for summarization.

**Clean content ingestion**: Convert a known documentation or article URL to LLM-friendly text before chunking or embedding.

**Fact verification**: Extract the page behind a cited URL so an agent can quote or cross-check the source text.
<!-- END MANUAL -->

---
