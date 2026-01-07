# Firecrawl Search

### What it is
Firecrawl searches the web for the given query.

### What it does
Firecrawl searches the web for the given query.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | The query to search for | str | Yes |
| limit | The number of pages to crawl | int | No |
| max_age | The maximum age of the page in milliseconds - default is 1 hour | int | No |
| wait_for | Specify a delay in milliseconds before fetching the content, allowing the page sufficient time to load. | int | No |
| formats | Returns the content of the search if specified | List["markdown" | "html" | "rawHtml"] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the search failed | str |
| data | The result of the search | Dict[str, True] |
| site | The site of the search | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
