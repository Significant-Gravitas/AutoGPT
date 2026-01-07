# Airtable Create Field

### What it is
Add a new field to an Airtable table.

### What it does
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
| field_type | The type of the field to create | "singleLineText" | "email" | "url" | No |
| name | The name of the field to create | str | Yes |
| description | The description of the field to create | str | No |
| options | The options of the field to create | Dict[str, str] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| field | Created field object | Dict[str, True] |
| field_id | ID of the created field | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Airtable Create Table

### What it is
Create a new table in an Airtable base.

### What it does
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
| table_fields | Table fields with name, type, and options | List[Dict[str, True]] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| table | Created table object | Dict[str, True] |
| table_id | ID of the created table | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Airtable List Schema

### What it is
Get the complete schema of an Airtable base.

### What it does
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
| base_schema | Complete base schema with tables, fields, and views | Dict[str, True] |
| tables | Array of table objects | List[Dict[str, True]] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Airtable Update Field

### What it is
Update field properties in an Airtable table.

### What it does
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
| field | Updated field object | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Airtable Update Table

### What it is
Update table properties.

### What it does
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
| date_dependency | The date dependency of the table to update | Dict[str, True] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| table | Updated table object | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
