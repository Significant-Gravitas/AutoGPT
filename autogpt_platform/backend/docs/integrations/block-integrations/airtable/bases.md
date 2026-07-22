# Airtable Bases
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Airtable Create Base

### What it is
Create or find a base in Airtable

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| workspace_id | The workspace ID where the base will be created | str | Yes |
| name | The name of the new base | str | Yes |
| find_existing | If true, return existing base with same name instead of creating duplicate | bool | No |
| tables | At least one table and field must be specified. Array of table objects to create in the base. Each table should have 'name' and 'fields' properties | List[Dict[str, Any]] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| base_id | The ID of the created or found base | str |
| tables | Array of table objects | List[Dict[str, Any]] |
| table | A single table object | Dict[str, Any] |
| was_created | True if a new base was created, False if existing was found | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Airtable List Bases

### What it is
List all bases in Airtable

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| trigger | Trigger the block to run - value is ignored | str | No |
| offset | Pagination offset from previous request | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| bases | Array of base objects | List[Dict[str, Any]] |
| offset | Offset for next page (null if no more bases) | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
