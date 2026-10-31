# Exa Websets Import Export
<!-- MANUAL: file_description -->
Blocks for importing and exporting data with Exa websets.
<!-- END MANUAL -->

## Exa Create Import

### What it is
Import CSV data to use with websets for targeted searches

### How it works
<!-- MANUAL: how_it_works -->
This block uploads CSV text to Exa as an import that webset searches can use as a scope (search within it) or exclude source, or that a new webset can copy items from. It treats the first row as a header, so it reports the number of rows minus one as the item count, along with the data size, the entity type and the 0-based column holding each item's identifier.

`url_column` is not part of Exa's current import API (Exa infers columns from the file) and is only sent if you set it. For a 'custom' entity type the description is sent only when you provide one. The block returns the import's initial status; use Get Import to check when it has finished processing. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| title | Title for this import | str | Yes |
| csv_data | CSV data to import (as a string) | str | Yes |
| entity_type | Type of entities being imported | "company" \| "person" \| "article" \| "research_paper" \| "custom" | No |
| entity_description | Description for custom entity type | str | No |
| identifier_column | Column index containing the identifier (0-based) | int | No |
| url_column | Column index containing URLs. Not part of Exa's current import API (Exa infers columns from the file); sent as-is if set. | int | No |
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

Use this to clean up imports that are no longer needed or contain outdated data. The deletion cannot be undone. It charges only if Exa's response reports a cost (100 credits per USD).
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
This block pages through the webset's items and returns up to `max_items` (1-100, default 100) as one string in JSON (an indented array), JSON Lines or CSV. CSV flattens nested fields into underscore-joined columns and writes lists as JSON text. Turning off `include_enrichments` removes each item's enrichments; `include_content` only removes a top-level `content` field, and item content sits inside `properties`, so it is still exported.

`total_items` is the number of items exported, not the size of the webset, and `truncated` is true whenever the export reached `max_items`, even if no items were left over. This block never charges, because the item listing it uses reports no cost.
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
This block sends `GET /v0/imports/{id}` and returns the import's status (pending, processing, completed, failed or canceled), title, format, entity type, item count, metadata and timestamps. Use this to check if an import is ready to use or to troubleshoot failed imports.

The upload URL and its expiry are filled in while the import is waiting for its data, and the failure reason and message when it failed. It charges only if Exa's response reports a cost (100 credits per USD).
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
| status | Current status: pending, processing, completed, failed or canceled | str |
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
This block sends `GET /v0/imports` and returns one page of imports (`limit` 1-200, default 25), both as a list and one at a time, with each import's title, status, entity type and item count. While `has_more` is true, pass `next_cursor` back in as `cursor` to get the next page.

Use this to discover existing imports that can be referenced in webset searches or to manage your import library. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| limit | Number of imports to return (1-200) | int | No |
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
