# Airtable Schema
<!-- MANUAL: file_description -->
Blocks for managing Airtable schema including tables, fields, and their configurations.
<!-- END MANUAL -->

## Airtable Create Field

### What it is
Add a new field to an Airtable table

### How it works
<!-- MANUAL: how_it_works -->
This block adds a new field to an existing Airtable table using the Airtable API. Specify the field type (text, email, URL, etc.), name, and optional description and configuration options.

The field is created immediately and becomes available for use in all records. Returns the created field object with its assigned ID.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| base_id | The Airtable base ID | str | Yes |
| table_id | The table ID to add field to | str | Yes |
| field_type | The type of the field to create | "singleLineText" \| "email" \| "url" \| "multilineText" \| "number" \| "percent" \| "currency" \| "singleSelect" \| "multipleSelects" \| "singleCollaborator" \| "multipleCollaborators" \| "multipleRecordLinks" \| "date" \| "dateTime" \| "phoneNumber" \| "multipleAttachments" \| "checkbox" \| "formula" \| "createdTime" \| "rollup" \| "count" \| "lookup" \| "multipleLookupValues" \| "autoNumber" \| "barcode" \| "rating" \| "richText" \| "duration" \| "lastModifiedTime" \| "button" \| "createdBy" \| "lastModifiedBy" \| "externalSyncSource" \| "aiText" | No |
| name | The name of the field to create | str | Yes |
| description | The description of the field to create | str | No |
| options | The options of the field to create | Dict[str, str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| field | Created field object | Dict[str, Any] |
| field_id | ID of the created field | str |

### Possible use case
<!-- MANUAL: use_case -->
**Schema Evolution**: Add new fields to tables as application requirements grow.

**Dynamic Forms**: Create fields based on user configuration or form builder settings.

**Data Integration**: Add fields to capture data from newly integrated external systems.
<!-- END MANUAL -->

---

## Airtable Create Table

### What it is
Create a new table in an Airtable base

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new table in an Airtable base with the specified name and optional field definitions. Each field definition includes name, type, and type-specific options.

The table is created with the defined schema and is immediately ready for use. Returns the created table object with its ID.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| base_id | The Airtable base ID | str | Yes |
| table_name | The name of the table to create | str | Yes |
| table_fields | Table fields with name, type, and options | List[Dict[str, Any]] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| table | Created table object | Dict[str, Any] |
| table_id | ID of the created table | str |

### Possible use case
<!-- MANUAL: use_case -->
**Application Scaffolding**: Create tables programmatically when setting up new application modules.

**Multi-Tenant Setup**: Generate customer-specific tables dynamically.

**Feature Expansion**: Add new tables as features are enabled or installed.
<!-- END MANUAL -->

---

## Airtable List Schema

### What it is
Get the complete schema of an Airtable base

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves the complete schema of an Airtable base, including all tables, their fields, field types, and views. This metadata is essential for building dynamic integrations that need to understand table structure.

The schema includes field configurations, validation rules, and relationship definitions between tables.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| base_id | The Airtable base ID | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| base_schema | Complete base schema with tables, fields, and views | Dict[str, Any] |
| tables | Array of table objects | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
**Schema Discovery**: Understand table structure for building dynamic forms or queries.

**Documentation**: Generate documentation of database schema automatically.

**Migration Planning**: Analyze schema before migrating data to other systems.
<!-- END MANUAL -->

---

## Airtable Update Field

### What it is
Update field properties in an Airtable table

### How it works
<!-- MANUAL: how_it_works -->
This block updates properties of an existing field in an Airtable table. You can modify the field name and description. Note that field type cannot be changed after creation.

Changes take effect immediately across all records and views that use the field.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| base_id | The Airtable base ID | str | Yes |
| table_id | The table ID containing the field | str | Yes |
| field_id | The field ID to update | str | Yes |
| name | The name of the field to update | str | No |
| description | The description of the field to update | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| field | Updated field object | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Field Renaming**: Update field names to match evolving terminology or standards.

**Documentation Updates**: Add or update field descriptions for better team understanding.

**Schema Maintenance**: Keep field metadata current as application requirements change.
<!-- END MANUAL -->

---

## Airtable Update Table

### What it is
Update table properties

### How it works
<!-- MANUAL: how_it_works -->
This block updates table properties in an Airtable base. You can change the table name, description, and date dependency settings. Changes apply immediately and affect all users accessing the table.

This is useful for maintaining table metadata and organizing your base structure.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| base_id | The Airtable base ID | str | Yes |
| table_id | The table ID to update | str | Yes |
| table_name | The name of the table to update | str | No |
| table_description | The description of the table to update | str | No |
| date_dependency | The date dependency of the table to update | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| table | Updated table object | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Table Organization**: Rename tables to follow naming conventions or reflect current usage.

**Description Management**: Update table descriptions for documentation purposes.

**Configuration Updates**: Modify table settings like date dependencies as requirements change.
<!-- END MANUAL -->

---
