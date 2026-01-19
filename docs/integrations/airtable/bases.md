# Airtable Bases
<!-- MANUAL: file_description -->
Blocks for creating and managing Airtable bases, which are the top-level containers for tables, records, and data in Airtable.
<!-- END MANUAL -->

## Airtable Create Base

### What it is
Create or find a base in Airtable

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new Airtable base in a specified workspace, or finds an existing one with the same name. When creating, you can optionally define initial tables and their fields to set up the schema.

Enable find_existing to search for a base with the same name before creating a new one, preventing duplicates in your workspace.
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
**Project Setup**: Automatically create new bases when projects start with predefined table structures.

**Template Deployment**: Deploy standardized base templates across teams or clients.

**Multi-Tenant Apps**: Create separate bases for each customer or project programmatically.
<!-- END MANUAL -->

---

## Airtable List Bases

### What it is
List all bases in Airtable

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves a list of all Airtable bases accessible to your connected account. It returns basic information about each base including ID, name, and permission level.

Results are paginated; use the offset output to retrieve additional pages if there are more bases than returned in a single call.
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
**Base Discovery**: Find available bases for building dynamic dropdowns or navigation.

**Inventory Management**: List all bases in an organization for auditing or documentation.

**Cross-Base Operations**: Enumerate bases to perform operations across multiple databases.
<!-- END MANUAL -->

---
