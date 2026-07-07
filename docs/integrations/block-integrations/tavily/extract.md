# Tavily Extract
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Tavily Extract

### What it is
Extracts page content from one or more URLs using Tavily, optimized for LLM consumption

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| urls | The URLs to extract content from (up to 20 per request) | List[str] | Yes |
| extract_depth | Depth of the extraction: basic or advanced (retrieves more data, including tables and embedded content) | "basic" \| "advanced" | No |
| format | The format of the extracted content | "markdown" \| "text" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the extraction failed | str |
| results | List of successfully extracted pages | List[TavilyPageContent] |
| result | Single extracted page | TavilyPageContent |
| failed_urls | URLs that could not be extracted | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
