# Exa Bulk Webset Items

### What it is
Get all items from a webset in bulk (with configurable limits).

### What it does
Get all items from a webset in bulk (with configurable limits)

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| max_items | Maximum number of items to retrieve (1-1000). Note: Large values may take longer. | int | No |
| include_enrichments | Include enrichment data for each item | bool | No |
| include_content | Include full content for each item | bool | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| items | All items from the webset | List[WebsetItemModel] |
| item | Individual item (yielded for each item) | WebsetItemModel |
| total_retrieved | Total number of items retrieved | int |
| truncated | Whether results were truncated due to max_items limit | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Delete Webset Item

### What it is
Delete a specific item from a webset.

### What it does
Delete a specific item from a webset

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| item_id | The ID of the item to delete | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| item_id | The ID of the deleted item | str |
| success | Whether the deletion was successful | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Get New Items

### What it is
Get items added since a cursor - enables incremental processing without reprocessing.

### What it does
Get items added since a cursor - enables incremental processing without reprocessing

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| since_cursor | Cursor from previous run - only items after this will be returned. Leave empty on first run. | str | No |
| max_items | Maximum number of new items to retrieve | int | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| new_items | Items added since the cursor | List[WebsetItemModel] |
| item | Individual item (yielded for each new item) | WebsetItemModel |
| count | Number of new items found | int |
| next_cursor | Save this cursor for the next run to get only newer items | str |
| has_more | Whether there are more new items beyond max_items | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Get Webset Item

### What it is
Get a specific item from a webset by its ID.

### What it does
Get a specific item from a webset by its ID

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| item_id | The ID of the specific item to retrieve | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| item_id | The unique identifier for the item | str |
| url | The URL of the original source | str |
| title | The title of the item | str |
| content | The main content of the item | str |
| entity_data | Entity-specific structured data | Dict[str, True] |
| enrichments | Enrichment data added to the item | Dict[str, True] |
| created_at | When the item was added to the webset | str |
| updated_at | When the item was last updated | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa List Webset Items

### What it is
List items in a webset with pagination support.

### What it does
List items in a webset with pagination support

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| limit | Number of items to return (1-100) | int | No |
| cursor | Cursor for pagination through results | str | No |
| wait_for_items | Wait for items to be available if webset is still processing | bool | No |
| wait_timeout | Maximum time to wait for items in seconds | int | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| items | List of webset items | List[WebsetItemModel] |
| webset_id | The ID of the webset | str |
| item | Individual item (yielded for each item in the list) | WebsetItemModel |
| has_more | Whether there are more items to paginate through | bool |
| next_cursor | Cursor for the next page of results | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Webset Items Summary

### What it is
Get a summary of webset items without retrieving all data.

### What it does
Get a summary of webset items without retrieving all data

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| sample_size | Number of sample items to include | int | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| total_items | Total number of items in the webset | int |
| entity_type | Type of entities in the webset | str |
| sample_items | Sample of items from the webset | List[WebsetItemModel] |
| enrichment_columns | List of enrichment columns available | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
