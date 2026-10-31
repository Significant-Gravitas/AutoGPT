# Exa Websets Items
<!-- MANUAL: file_description -->
Blocks for retrieving and managing items within Exa websets.
<!-- END MANUAL -->

## Exa Bulk Webset Items

### What it is
Get all items from a webset in bulk (with configurable limits)

### How it works
<!-- MANUAL: how_it_works -->
This block pages through the webset's items, up to 100 per page, until it has `max_items` (1-1000, default 100) or there are no more, and returns them as a list and one at a time. Turning off `include_enrichments` or `include_content` empties those fields in the output; it does not make the requests smaller.

`truncated` is true whenever the block reached `max_items`, even if no items were left over. This block never charges, because the item listing it uses reports no cost.
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

Use this to clean up irrelevant results, remove duplicates, or curate webset contents by removing items that don't meet your quality standards. It charges only if Exa's response reports a cost (100 credits per USD).
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
This block lists the webset's items after `since_cursor` (from the start when it is empty), up to 100 per page, until it has `max_items` (1-1000, default 100) or no more remain. It returns the items, their count, `has_more`, and the `next_cursor` to save for the next run.

On the last page Exa can return an empty `next_cursor`; passing that in starts again from the first item, so keep your previous cursor in that case. It charges only if Exa's response reports a cost (100 credits per USD).
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
This block retrieves one item from a webset and returns its URL, title, content, entity data, enrichment results (keyed by enrichment ID) and timestamps. The title is the person's or company's name, or the item description for other entity types; content is empty for people.

Use this when you need complete data for a particular item. It charges only if Exa's response reports a cost (100 credits per USD).
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
This block returns one page of a webset's items (`limit` 1-100, default 25), optionally only those that came from one search or import (`source_id`). With `wait_for_items` on, it checks again every 2 seconds (backing off to 10) until the page has items or `wait_timeout` (default 60, max 300) passes, then returns what it has, possibly nothing, without failing.

While `has_more` is true, pass `next_cursor` back in as `cursor` to get the next page. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| limit | Number of items to return (1-100) | int | No |
| cursor | Cursor for pagination through results | str | No |
| source_id | Only return items that came from this search or import ID | str | No |
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
This block reads the webset and returns its entity type (taken from the first search), its enrichment column titles, and up to `sample_size` (0-10, default 5) sample items. `total_items` is the total found by the webset's searches, because Exa reports no webset-wide total, so imported items are not counted.

Use this to understand webset contents at a glance, check enrichment availability, or get sample data for validation. It charges only if Exa's response reports a cost (100 credits per USD).
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
