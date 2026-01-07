# Notion Search

### What it is
Search your Notion workspace for pages and databases by text query.

### What it does
Search your Notion workspace for pages and databases by text query.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query text. Leave empty to get all accessible pages/databases. | str | No |
| filter_type | Filter results by type: 'page' or 'database'. Leave empty for both. | str | No |
| limit | Maximum number of results to return | int | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| results | List of search results with title, type, URL, and metadata. | List[NotionSearchResult] |
| result | Individual search result (yields one per result found). | NotionSearchResult |
| result_ids | List of IDs from search results for batch operations. | List[str] |
| count | Number of results found. | int |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
