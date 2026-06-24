# Airtable Schema
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Airtable Create Field

### What it is
Add a new field to an Airtable table

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Airtable Create Table

### What it is
Create a new table in an Airtable base

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Airtable List Schema

### What it is
Get the complete schema of an Airtable base

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Airtable Update Field

### What it is
Update field properties in an Airtable table

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Airtable Update Table

### What it is
Update table properties

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
