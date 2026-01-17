# Notion Read Database
<!-- MANUAL: file_description -->
Blocks for querying and reading data from Notion databases.
<!-- END MANUAL -->

## Notion Read Database

### What it is
Query a Notion database with optional filtering and sorting, returning structured entries.

### How it works
<!-- MANUAL: how_it_works -->
This block queries a Notion database using the Notion API. It retrieves entries with optional filtering by property values and sorting. The block requires your Notion integration to have access to the database.

Results include all property values for each entry, the entry IDs for further operations, and the total count. The database connection must be shared with your integration from Notion.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| database_id | Notion database ID. Must be accessible by the connected integration. | str | Yes |
| filter_property | Property name to filter by (e.g., 'Status', 'Priority') | str | No |
| filter_value | Value to filter for in the specified property | str | No |
| sort_property | Property name to sort by | str | No |
| sort_direction | Sort direction: 'ascending' or 'descending' | str | No |
| limit | Maximum number of entries to retrieve | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| entries | List of database entries with their properties. | List[Dict[str, Any]] |
| entry | Individual database entry (yields one per entry found). | Dict[str, Any] |
| entry_ids | List of entry IDs for batch operations. | List[str] |
| entry_id | Individual entry ID (yields one per entry found). | str |
| count | Number of entries retrieved. | int |
| database_title | Title of the database. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Task Management**: Query a Notion task database to find items with a specific status or assigned to a particular person.

**Content Pipeline**: Read entries from a content calendar database to identify posts scheduled for today or this week.

**CRM Sync**: Fetch customer records from a Notion database to sync with other systems or trigger workflows.
<!-- END MANUAL -->

---
