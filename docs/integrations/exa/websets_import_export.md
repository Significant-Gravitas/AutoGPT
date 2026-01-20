# Exa Websets Import Export
<!-- MANUAL: file_description -->
Blocks for importing and exporting data with Exa websets.
<!-- END MANUAL -->

## Exa Create Import

### What it is
Import CSV data to use with websets for targeted searches

### How it works
<!-- MANUAL: how_it_works -->
This block creates an import from CSV data that can be used as a source for webset searches. Imports allow you to bring your own data (like company lists or contact lists) and use them for scoped or exclusion searches.

You specify the entity type and which columns contain identifiers and URLs. The import becomes available as a source that can be referenced when creating webset searches.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| title | Title for this import | str | Yes |
| csv_data | CSV data to import (as a string) | str | Yes |
| entity_type | Type of entities being imported | "company" \| "person" \| "article" \| "research_paper" \| "custom" | No |
| entity_description | Description for custom entity type | str | No |
| identifier_column | Column index containing the identifier (0-based) | int | No |
| url_column | Column index containing URLs (optional) | int | No |
| metadata | Metadata to attach to the import | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| import_id | The unique identifier for the created import | str |
| status | Current status of the import | str |
| title | Title of the import | str |
| count | Number of items in the import | int |
| entity_type | Type of entities imported | str |
| upload_url | Upload URL for CSV data (only if csv_data not provided in request) | str |
| upload_valid_until | Expiration time for upload URL (only if upload_url is provided) | str |
| created_at | When the import was created | str |

### Possible use case
<!-- MANUAL: use_case -->
**Customer Enrichment**: Import your customer list to find similar companies or related contacts.

**Exclusion Lists**: Import existing leads to exclude from new prospecting searches.

**Targeted Expansion**: Use imported data as a starting point for relationship-based searches.
<!-- END MANUAL -->

---

## Exa Delete Import

### What it is
Delete an import

### How it works
<!-- MANUAL: how_it_works -->
This block permanently deletes an import and its data. Any websets that reference this import for scoped or exclusion searches will no longer have access to it.

Use this to clean up imports that are no longer needed or contain outdated data. The deletion cannot be undone.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| import_id | The ID of the import to delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| import_id | The ID of the deleted import | str |
| success | Whether the deletion was successful | str |

### Possible use case
<!-- MANUAL: use_case -->
**Data Refresh**: Delete outdated imports before uploading updated versions.

**Cleanup Operations**: Remove imports that are no longer used in any webset searches.

**Compliance**: Delete imports containing data that needs to be removed for privacy compliance.
<!-- END MANUAL -->

---

## Exa Export Webset

### What it is
Export webset data in JSON, CSV, or JSON Lines format

### How it works
<!-- MANUAL: how_it_works -->
This block exports all items from a webset in your chosen format. You can include full content and enrichment data in the export, and limit the number of items exported.

Supported formats include JSON for structured data, CSV for spreadsheet compatibility, and JSON Lines for streaming or large dataset processing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to export | str | Yes |
| format | Export format | "json" \| "csv" \| "jsonl" | No |
| include_content | Include full content in export | bool | No |
| include_enrichments | Include enrichment data in export | bool | No |
| max_items | Maximum number of items to export | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| export_data | Exported data in the requested format | str |
| item_count | Number of items exported | int |
| total_items | Total number of items in the webset | int |
| truncated | Whether the export was truncated due to max_items limit | bool |
| format | Format of the exported data | str |

### Possible use case
<!-- MANUAL: use_case -->
**CRM Integration**: Export webset data as CSV to import into CRM or marketing automation systems.

**Reporting**: Generate exports for analysis in spreadsheets or business intelligence tools.

**Backup**: Create periodic exports of valuable webset data for archival purposes.
<!-- END MANUAL -->

---

## Exa Get Import

### What it is
Get the status and details of an import

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves detailed information about an import including its status, item count, and configuration. Use this to check if an import is ready to use or to troubleshoot failed imports.

The block returns upload status information if the import is pending data upload, or failure details if the import encountered errors.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| import_id | The ID of the import to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| import_id | The unique identifier for the import | str |
| status | Current status of the import | str |
| title | Title of the import | str |
| format | Format of the imported data | str |
| entity_type | Type of entities imported | str |
| count | Number of items imported | int |
| upload_url | Upload URL for CSV data (if import not yet uploaded) | str |
| upload_valid_until | Expiration time for upload URL (if applicable) | str |
| failed_reason | Reason for failure (if applicable) | str |
| failed_message | Detailed failure message (if applicable) | str |
| created_at | When the import was created | str |
| updated_at | When the import was last updated | str |
| metadata | Metadata attached to the import | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Status Verification**: Check import status after upload to confirm data is ready for use.

**Error Investigation**: Retrieve import details to understand why an import failed.

**Audit Trail**: Review import configuration and metadata for documentation purposes.
<!-- END MANUAL -->

---

## Exa List Imports

### What it is
List all imports with pagination support

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves a paginated list of all your imports. Results include basic information about each import such as title, status, and item count.

Use this to discover existing imports that can be referenced in webset searches or to manage your import library.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| limit | Number of imports to return | int | No |
| cursor | Cursor for pagination | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| imports | List of imports | List[Dict[str, Any]] |
| import_item | Individual import (yielded for each import) | Dict[str, Any] |
| has_more | Whether there are more imports to paginate through | bool |
| next_cursor | Cursor for the next page of results | str |

### Possible use case
<!-- MANUAL: use_case -->
**Import Discovery**: Find existing imports to reference when creating new webset searches.

**Library Management**: Review all imports to identify outdated data that can be cleaned up.

**Source Selection**: Browse available imports when setting up scoped or exclusion searches.
<!-- END MANUAL -->

---
