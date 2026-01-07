# Notion Read Database

### What it is
Query a Notion database with optional filtering and sorting, returning structured entries.

### What it does
Query a Notion database with optional filtering and sorting, returning structured entries.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| entries | List of database entries with their properties. | List[Dict[str, True]] |
| entry | Individual database entry (yields one per entry found). | Dict[str, True] |
| entry_ids | List of entry IDs for batch operations. | List[str] |
| entry_id | Individual entry ID (yields one per entry found). | str |
| count | Number of entries retrieved. | int |
| database_title | Title of the database. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
