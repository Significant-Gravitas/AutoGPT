# Anysearch Extract
<!-- MANUAL: file_description -->
Extracts readable page content from a URL via AnySearch. Requires an `ANYSEARCH_API_KEY` credential - see the AnySearch Search page for setup.
<!-- END MANUAL -->

## Any Search Extract

### What it is
Extracts the content of a single URL as markdown using AnySearch, optimized for LLM consumption

### How it works
<!-- MANUAL: how_it_works -->
The block posts the single URL to `POST https://api.anysearch.com/v1/extract` and returns the page title plus body content as markdown. One URL is extracted per run - the REST surface has no batch-extract endpoint, so fan URLs out across the graph when you need several. Transport, HTTP and API-level failures raise a block error rather than emitting partial output.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL to extract content from | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the extraction failed | str |
| url | URL of the extracted page | str |
| title | Title of the extracted page | str |
| content | Extracted page content, formatted as markdown | str |

### Possible use case
<!-- MANUAL: use_case -->
**Deep-read a search hit**: Pipe the url output of the AnySearch Search block into this block to turn a ranked result into full markdown for summarization.

**Clean content ingestion**: Convert a known documentation or article URL to LLM-friendly markdown before chunking or embedding.

**Fact verification**: Extract the page behind a cited URL so an agent can quote or cross-check the source text.
<!-- END MANUAL -->

---
