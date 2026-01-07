# Exa Cancel Enrichment

### What it is
Cancel a running enrichment operation.

### What it does
Cancel a running enrichment operation

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| enrichment_id | The ID of the enrichment to cancel | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| enrichment_id | The ID of the canceled enrichment | str |
| status | Status after cancellation | str |
| items_enriched_before_cancel | Approximate number of items enriched before cancellation | int |
| success | Whether the cancellation was successful | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Create Enrichment

### What it is
Create enrichments to extract additional structured data from webset items.

### What it does
Create enrichments to extract additional structured data from webset items

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| description | What data to extract from each item | str | Yes |
| title | Short title for this enrichment (auto-generated if not provided) | str | No |
| format | Expected format of the extracted data | "text" | "date" | "number" | No |
| options | Available options when format is 'options' | List[str] | No |
| apply_to_existing | Apply this enrichment to existing items in the webset | bool | No |
| metadata | Metadata to attach to the enrichment | Dict[str, True] | No |
| wait_for_completion | Wait for the enrichment to complete on existing items | bool | No |
| polling_timeout | Maximum time to wait for completion in seconds | int | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| enrichment_id | The unique identifier for the created enrichment | str |
| webset_id | The webset this enrichment belongs to | str |
| status | Current status of the enrichment | str |
| title | Title of the enrichment | str |
| description | Description of what data is extracted | str |
| format | Format of the extracted data | str |
| instructions | Generated instructions for the enrichment | str |
| items_enriched | Number of items enriched (if wait_for_completion was True) | int |
| completion_time | Time taken to complete in seconds (if wait_for_completion was True) | float |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Delete Enrichment

### What it is
Delete an enrichment from a webset.

### What it does
Delete an enrichment from a webset

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| enrichment_id | The ID of the enrichment to delete | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| enrichment_id | The ID of the deleted enrichment | str |
| success | Whether the deletion was successful | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Get Enrichment

### What it is
Get the status and details of a webset enrichment.

### What it does
Get the status and details of a webset enrichment

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| enrichment_id | The ID of the enrichment to retrieve | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| enrichment_id | The unique identifier for the enrichment | str |
| status | Current status of the enrichment | str |
| title | Title of the enrichment | str |
| description | Description of what data is extracted | str |
| format | Format of the extracted data | str |
| options | Available options (for 'options' format) | List[str] |
| instructions | Generated instructions for the enrichment | str |
| created_at | When the enrichment was created | str |
| updated_at | When the enrichment was last updated | str |
| metadata | Metadata attached to the enrichment | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Update Enrichment

### What it is
Update an existing enrichment configuration.

### What it does
Update an existing enrichment configuration

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| enrichment_id | The ID of the enrichment to update | str | Yes |
| description | New description for what data to extract | str | No |
| format | New format for the extracted data | "text" | "date" | "number" | No |
| options | New options when format is 'options' | List[str] | No |
| metadata | New metadata to attach to the enrichment | Dict[str, True] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| enrichment_id | The unique identifier for the enrichment | str |
| status | Current status of the enrichment | str |
| title | Title of the enrichment | str |
| description | Updated description | str |
| format | Updated format | str |
| success | Whether the update was successful | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
