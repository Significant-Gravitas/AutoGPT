# Firecrawl Extract

### What it is
Firecrawl crawls websites to extract comprehensive data while bypassing blockers.

### What it does
Firecrawl crawls websites to extract comprehensive data while bypassing blockers.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| urls | The URLs to crawl - at least one is required. Wildcards are supported. (/*) | List[str] | Yes |
| prompt | The prompt to use for the crawl | str | No |
| output_schema | A Json Schema describing the output structure if more rigid structure is desired. | Dict[str, True] | No |
| enable_web_search | When true, extraction can follow links outside the specified domain. | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the extraction failed | str |
| data | The result of the crawl | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
