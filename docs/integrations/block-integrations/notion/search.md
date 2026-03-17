# Notion Search
<!-- MANUAL: file_description -->
Blocks for searching pages and databases in a Notion workspace.
<!-- END MANUAL -->

## Notion Search

### What it is
Search your Notion workspace for pages and databases by text query.

### How it works
<!-- MANUAL: how_it_works -->
This block searches across your Notion workspace using the Notion Search API. It finds pages and databases matching your query text, with optional filtering by type (page or database).

Results include titles, types, URLs, and metadata for each match. Leave the query empty to retrieve all accessible pages and databases. Pagination is handled automatically up to the specified limit.
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
**Content Discovery**: Find relevant pages in your workspace based on keywords or topics.

**Database Lookup**: Search for specific databases to use in subsequent operations.

**Knowledge Retrieval**: Search your Notion workspace to find answers or related documentation.
<!-- END MANUAL -->

---
