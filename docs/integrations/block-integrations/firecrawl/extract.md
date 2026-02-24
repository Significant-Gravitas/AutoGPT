# Firecrawl Extract
<!-- MANUAL: file_description -->
Blocks for extracting structured data from web pages using Firecrawl's AI extraction.
<!-- END MANUAL -->

## Firecrawl Extract

### What it is
Firecrawl crawls websites to extract comprehensive data while bypassing blockers.

### How it works
<!-- MANUAL: how_it_works -->
This block uses Firecrawl's extraction API to pull structured data from web pages based on a prompt or schema. It crawls the specified URLs and uses AI to extract information matching your requirements.

Define the data structure you want using a JSON schema for precise extraction, or use natural language prompts for flexible extraction. Wildcards in URLs allow extracting data from multiple pages matching a pattern.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| urls | The URLs to crawl - at least one is required. Wildcards are supported. (/*) | List[str] | Yes |
| prompt | The prompt to use for the crawl | str | No |
| output_schema | A Json Schema describing the output structure if more rigid structure is desired. | Dict[str, Any] | No |
| enable_web_search | When true, extraction can follow links outside the specified domain. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the extraction failed | str |
| data | The result of the crawl | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Product Data Extraction**: Extract structured product information (prices, specs, reviews) from e-commerce sites.

**Contact Scraping**: Pull business contact information from company websites in a structured format.

**Data Pipeline Input**: Automatically extract and structure web data for analysis or database population.
<!-- END MANUAL -->

---
