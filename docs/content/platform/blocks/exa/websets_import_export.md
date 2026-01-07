# Exa Create Import

### What it is
Import CSV data to use with websets for targeted searches.

### What it does
Import CSV data to use with websets for targeted searches

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| title | Title for this import | str | Yes |
| csv_data | CSV data to import (as a string) | str | Yes |
| entity_type | Type of entities being imported | "company" | "person" | "article" | No |
| entity_description | Description for custom entity type | str | No |
| identifier_column | Column index containing the identifier (0-based) | int | No |
| url_column | Column index containing URLs (optional) | int | No |
| metadata | Metadata to attach to the import | Dict[str, True] | No |

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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Delete Import

### What it is
Delete an import.

### What it does
Delete an import

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Export Webset

### What it is
Export webset data in JSON, CSV, or JSON Lines format.

### What it does
Export webset data in JSON, CSV, or JSON Lines format

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to export | str | Yes |
| format | Export format | "json" | "csv" | "jsonl" | No |
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Get Import

### What it is
Get the status and details of an import.

### What it does
Get the status and details of an import

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| metadata | Metadata attached to the import | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa List Imports

### What it is
List all imports with pagination support.

### What it does
List all imports with pagination support

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| imports | List of imports | List[Dict[str, True]] |
| import_item | Individual import (yielded for each import) | Dict[str, True] |
| has_more | Whether there are more imports to paginate through | bool |
| next_cursor | Cursor for the next page of results | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
