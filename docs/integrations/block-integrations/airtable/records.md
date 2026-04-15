# Airtable Records
<!-- MANUAL: file_description -->
Blocks for creating, reading, updating, and deleting records in Airtable tables.
<!-- END MANUAL -->

## Airtable Create Records

### What it is
Create records in an Airtable table

### How it works
<!-- MANUAL: how_it_works -->
This block creates new records in an Airtable table using the Airtable API. Each record is specified with a fields object containing field names and values. You can create up to 10 records in a single call.

Enable typecast to automatically convert string values to appropriate field types (dates, numbers, etc.). The block returns the created records with their assigned IDs.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| base_id | The Airtable base ID | str | Yes |
| table_id_or_name | Table ID or name | str | Yes |
| records | Array of records to create (each with 'fields' object) | List[Dict[str, Any]] | Yes |
| skip_normalization | Skip output normalization to get raw Airtable response (faster but may have missing fields) | bool | No |
| typecast | Automatically convert string values to appropriate types | bool | No |
| return_fields_by_field_id | Return fields by field ID | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| records | Array of created record objects | List[Dict[str, Any]] |
| details | Details of the created records | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Data Import**: Bulk import data from external sources into Airtable tables.

**Form Submissions**: Create records from form submissions or API integrations.

**Workflow Output**: Save workflow results or processed data to Airtable for tracking.
<!-- END MANUAL -->

---

## Airtable Delete Records

### What it is
Delete records from an Airtable table

### How it works
<!-- MANUAL: how_it_works -->
This block deletes records from an Airtable table by their record IDs. You can delete up to 10 records in a single call. The operation is permanent and cannot be undone.

Provide an array of record IDs to delete. Using the table ID instead of the name is recommended for reliability.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| base_id | The Airtable base ID | str | Yes |
| table_id_or_name | Table ID or name - It's better to use the table ID instead of the name | str | Yes |
| record_ids | Array of upto 10 record IDs to delete | List[str] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| records | Array of deletion results | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
**Data Cleanup**: Remove outdated or duplicate records from tables.

**Workflow Cleanup**: Delete temporary records after processing is complete.

**Batch Removal**: Remove multiple records that match certain criteria.
<!-- END MANUAL -->

---

## Airtable Get Record

### What it is
Get a single record from Airtable

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves a single record from an Airtable table by its ID. The record includes all field values and metadata like creation time. Enable normalize_output to ensure all fields are included with proper empty values.

Optionally include field metadata for type information and configuration details about each field.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| base_id | The Airtable base ID | str | Yes |
| table_id_or_name | Table ID or name | str | Yes |
| record_id | The record ID to retrieve | str | Yes |
| normalize_output | Normalize output to include all fields with proper empty values (disable to skip schema fetch and get raw Airtable response) | bool | No |
| include_field_metadata | Include field type and configuration metadata (requires normalize_output=true) | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | The record ID | str |
| fields | The record fields | Dict[str, Any] |
| created_time | The record created time | str |
| field_metadata | Field type and configuration metadata (only when include_field_metadata=true) | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Detail View**: Fetch complete record data for display or detailed processing.

**Record Lookup**: Retrieve specific records by ID from webhook payloads or references.

**Data Validation**: Check record contents before performing updates or related operations.
<!-- END MANUAL -->

---

## Airtable List Records

### What it is
List records from an Airtable table

### How it works
<!-- MANUAL: how_it_works -->
This block queries records from an Airtable table with optional filtering, sorting, and pagination. Use Airtable formulas to filter records and specify sort order by field and direction.

Results can be limited, paginated with offsets, and restricted to specific fields. Enable normalize_output for consistent field values across records.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| base_id | The Airtable base ID | str | Yes |
| table_id_or_name | Table ID or name | str | Yes |
| filter_formula | Airtable formula to filter records | str | No |
| view | View ID or name to use | str | No |
| sort | Sort configuration (array of {field, direction}) | List[Dict[str, Any]] | No |
| max_records | Maximum number of records to return | int | No |
| page_size | Number of records per page (max 100) | int | No |
| offset | Pagination offset from previous request | str | No |
| return_fields | Specific fields to return (comma-separated) | List[str] | No |
| normalize_output | Normalize output to include all fields with proper empty values (disable to skip schema fetch and get raw Airtable response) | bool | No |
| include_field_metadata | Include field type and configuration metadata (requires normalize_output=true) | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| records | Array of record objects | List[Dict[str, Any]] |
| offset | Offset for next page (null if no more records) | str |
| field_metadata | Field type and configuration metadata (only when include_field_metadata=true) | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Report Generation**: Query records with filters to build reports or dashboards.

**Data Export**: Fetch records matching criteria for export to other systems.

**Batch Processing**: List records to process in subsequent workflow steps.
<!-- END MANUAL -->

---

## Airtable Update Records

### What it is
Update records in an Airtable table

### How it works
<!-- MANUAL: how_it_works -->
This block updates existing records in an Airtable table. Each record update requires the record ID and a fields object with the values to update. Only specified fields are modified; other fields remain unchanged.

Enable typecast to automatically convert string values to appropriate types. You can update up to 10 records per call.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| base_id | The Airtable base ID | str | Yes |
| table_id_or_name | Table ID or name - It's better to use the table ID instead of the name | str | Yes |
| records | Array of records to update (each with 'id' and 'fields') | List[Dict[str, Any]] | Yes |
| typecast | Automatically convert string values to appropriate types | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| records | Array of updated record objects | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
**Status Updates**: Update record status fields as workflows progress.

**Data Enrichment**: Add computed or fetched data to existing records.

**Batch Modifications**: Update multiple records based on processed results.
<!-- END MANUAL -->

---
