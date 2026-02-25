# Exa Websets Items
<!-- MANUAL: file_description -->
Blocks for retrieving and managing items within Exa websets.
<!-- END MANUAL -->

## Exa Bulk Webset Items

### What it is
Get all items from a webset in bulk (with configurable limits)

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves all items from a webset in a single operation, automatically handling pagination internally. You can specify a maximum number of items and choose whether to include enrichments and full content.

Use this for batch processing when you need all webset data at once rather than paginating through results manually.
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
**Batch Processing**: Retrieve all webset items for bulk analysis or processing in external systems.

**Data Export**: Get complete webset data for integration with other tools or databases.

**Full Dataset Analysis**: Analyze entire webset contents when pagination isn't practical.
<!-- END MANUAL -->

---

## Exa Delete Webset Item

### What it is
Delete a specific item from a webset

### How it works
<!-- MANUAL: how_it_works -->
This block permanently removes a specific item from a webset. The item and all its enrichment data are deleted and cannot be recovered.

Use this to clean up irrelevant results, remove duplicates, or curate webset contents by removing items that don't meet your quality standards.
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
**Data Curation**: Remove irrelevant or low-quality items to improve webset accuracy.

**Duplicate Removal**: Delete duplicate entries discovered during review.

**Compliance**: Remove items that shouldn't be included for legal or policy reasons.
<!-- END MANUAL -->

---

## Exa Get New Items

### What it is
Get items added since a cursor - enables incremental processing without reprocessing

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves only items added to a webset since your last check, identified by a cursor. This enables efficient incremental processing without re-fetching previously processed items.

Save the returned next_cursor for subsequent calls to implement continuous incremental processing of new webset additions.
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
**Incremental Processing**: Process only new webset items in scheduled workflows without duplicating work.

**Real-Time Pipelines**: Build efficient pipelines that react to new data without full dataset scans.

**Change Detection**: Track what's new in websets for alerting or notification systems.
<!-- END MANUAL -->

---

## Exa Get Webset Item

### What it is
Get a specific item from a webset by its ID

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves detailed information about a specific webset item including its content, entity data, and enrichments. Use this when you need complete data for a particular item.

The block returns the full item record with all available data, timestamps, and any enrichment results that have been applied.
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
| entity_data | Entity-specific structured data | Dict[str, Any] |
| enrichments | Enrichment data added to the item | Dict[str, Any] |
| created_at | When the item was added to the webset | str |
| updated_at | When the item was last updated | str |

### Possible use case
<!-- MANUAL: use_case -->
**Detail View**: Fetch complete item data for display in detail views or profiles.

**Enrichment Review**: Retrieve item with enrichments to verify data extraction quality.

**Reference Lookup**: Get specific items by ID for cross-referencing or validation.
<!-- END MANUAL -->

---

## Exa List Webset Items

### What it is
List items in a webset with pagination support

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves a paginated list of items from a webset. You control page size and can optionally wait for items if the webset is still processing.

Use pagination cursors to iterate through large websets efficiently. Each page returns items along with metadata about whether more pages exist.
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
**Paginated Display**: Build UIs that display webset items with pagination controls.

**Streaming Processing**: Process webset items in manageable batches to avoid memory issues.

**Controlled Iteration**: Step through large websets methodically for thorough analysis.
<!-- END MANUAL -->

---

## Exa Webset Items Summary

### What it is
Get a summary of webset items without retrieving all data

### How it works
<!-- MANUAL: how_it_works -->
This block provides a lightweight summary of webset items including total count, entity type, available enrichment columns, and optional sample items. It's efficient for getting an overview without fetching full data.

Use this to understand webset contents at a glance, check enrichment availability, or get sample data for validation.
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
**Quick Overview**: Get webset statistics and samples without loading all data.

**Schema Discovery**: Check what enrichment columns are available before building exports.

**Validation**: Review sample items to verify webset quality before full processing.
<!-- END MANUAL -->

---
