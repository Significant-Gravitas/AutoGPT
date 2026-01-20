# Exa Websets
<!-- MANUAL: file_description -->
Blocks for creating and managing Exa websets for continuous web monitoring.
<!-- END MANUAL -->

## Exa Cancel Webset

### What it is
Cancel all operations being performed on a Webset

### How it works
<!-- MANUAL: how_it_works -->
This block cancels all running operations (searches, enrichments) on a webset. The webset transitions to an idle state and any in-progress operations are stopped.

The block is useful for stopping long-running operations that are no longer needed or when you need to modify the webset configuration. Items already processed before cancellation are retained.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to cancel | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webset_id | The unique identifier for the webset | str |
| status | The status of the webset after cancellation | str |
| external_id | The external identifier for the webset | str |
| success | Whether the cancellation was successful | str |

### Possible use case
<!-- MANUAL: use_case -->
**Resource Management**: Stop expensive operations on websets that are no longer needed.

**Configuration Updates**: Cancel operations before making changes to webset settings.

**Error Recovery**: Stop problematic operations and restart with corrected parameters.
<!-- END MANUAL -->

---

## Exa Create Or Find Webset

### What it is
Create a new webset or return existing one by external_id (idempotent operation)

### How it works
<!-- MANUAL: how_it_works -->
This block implements idempotent webset creation using an external ID. If a webset with the given external_id already exists, it returns that webset. Otherwise, it creates a new one.

This pattern prevents duplicate websets when workflows retry or run multiple times. The block indicates whether the webset was newly created or already existed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| external_id | External identifier for this webset - used to find existing or create new | str | Yes |
| search_query | Search query (optional - only needed if creating new webset) | str | No |
| search_count | Number of items to find in initial search | int | No |
| metadata | Key-value pairs to associate with the webset | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webset | The webset (existing or newly created) | Webset |
| was_created | True if webset was newly created, False if it already existed | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Idempotent Workflows**: Safely re-run workflows without creating duplicate websets.

**External System Integration**: Map websets to IDs from your own systems for easy reference.

**Retry-Safe Operations**: Handle workflow retries gracefully by reusing existing websets.
<!-- END MANUAL -->

---

## Exa Create Webset

### What it is
Create a new Exa Webset for persistent web search collections with optional waiting for initial results

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new Exa Webset, a persistent collection that stores web search results. You define a search query, entity type, and optional criteria that items must meet. The webset continuously evaluates potential matches against your criteria.

The block supports advanced features like scoped searches (searching within specific imports or other websets), enrichments for extracting structured data, and relationship-based "hop" searches. You can wait for initial results or return immediately for asynchronous processing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| search_query | Your search query. Use this to describe what you are looking for. Any URL provided will be crawled and used as context for the search. | str | Yes |
| search_count | Number of items the search will attempt to find. The actual number of items found may be less than this number depending on the search complexity. | int | No |
| search_entity_type | Entity type: 'company', 'person', 'article', 'research_paper', or 'custom'. If not provided, we automatically detect the entity from the query. | "company" \| "person" \| "article" \| "research_paper" \| "custom" \| "auto" | No |
| search_entity_description | Description for custom entity type (required when search_entity_type is 'custom') | str | No |
| search_criteria | List of criteria descriptions that every item will be evaluated against. If not provided, we automatically detect the criteria from the query. | List[str] | No |
| search_exclude_sources | List of source IDs (imports or websets) to exclude from search results | List[str] | No |
| search_exclude_types | List of source types corresponding to exclude sources ('import' or 'webset') | List["import" \| "webset"] | No |
| search_scope_sources | List of source IDs (imports or websets) to limit search scope to | List[str] | No |
| search_scope_types | List of source types corresponding to scope sources ('import' or 'webset') | List["import" \| "webset"] | No |
| search_scope_relationships | List of relationship definitions for hop searches (optional, one per scope source) | List[str] | No |
| search_scope_relationship_limits | List of limits on the number of related entities to find (optional, one per scope relationship) | List[int] | No |
| import_sources | List of source IDs to import from | List[str] | No |
| import_types | List of source types corresponding to import sources ('import' or 'webset') | List["import" \| "webset"] | No |
| enrichment_descriptions | List of enrichment task descriptions to perform on each webset item | List[str] | No |
| enrichment_formats | List of formats for enrichment responses ('text', 'date', 'number', 'options', 'email', 'phone'). If not specified, we automatically select the best format. | List["text" \| "date" \| "number" \| "options" \| "email" \| "phone"] | No |
| enrichment_options | List of option lists for enrichments with 'options' format. Each inner list contains the option labels. | List[List[str]] | No |
| enrichment_metadata | List of metadata dictionaries for enrichments | List[Dict[str, Any]] | No |
| external_id | External identifier for the webset. You can use this to reference the webset by your own internal identifiers. | str | No |
| metadata | Key-value pairs to associate with this webset | Dict[str, Any] | No |
| wait_for_initial_results | Wait for the initial search to complete before returning. This ensures you get results immediately. | bool | No |
| polling_timeout | Maximum time to wait for completion in seconds (only used if wait_for_initial_results is True) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webset | The created webset with full details | Webset |
| initial_item_count | Number of items found in the initial search (only if wait_for_initial_results was True) | int |
| completion_time | Time taken to complete the initial search in seconds (only if wait_for_initial_results was True) | float |

### Possible use case
<!-- MANUAL: use_case -->
**Lead Generation**: Create websets to find companies or people matching specific criteria for sales outreach.

**Competitive Intelligence**: Build persistent collections tracking competitors, market entrants, or industry news.

**Research Databases**: Compile curated collections of articles, papers, or resources on specific topics.
<!-- END MANUAL -->

---

## Exa Delete Webset

### What it is
Delete a Webset and all its items

### How it works
<!-- MANUAL: how_it_works -->
This block permanently deletes a webset and all of its items, searches, enrichments, and monitors. The operation cannot be undone.

Use this to clean up websets that are no longer needed or to remove test data. The block accepts either the Exa-generated ID or your custom external_id.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webset_id | The unique identifier for the deleted webset | str |
| external_id | The external identifier for the deleted webset | str |
| status | The status of the deleted webset | str |
| success | Whether the deletion was successful | str |

### Possible use case
<!-- MANUAL: use_case -->
**Cleanup Operations**: Remove completed or abandoned websets to maintain organization.

**Data Management**: Delete websets containing outdated or irrelevant data.

**Cost Control**: Remove unused websets to prevent unnecessary storage costs.
<!-- END MANUAL -->

---

## Exa Get Webset

### What it is
Retrieve a Webset by ID or external ID

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves detailed information about a specific webset including its status, configured searches, enrichments, and monitors.

The block returns the webset's current state, metadata, and timestamps. Use this to check webset configuration or monitor status before performing operations.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to retrieve | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webset_id | The unique identifier for the webset | str |
| status | The status of the webset | str |
| external_id | The external identifier for the webset | str |
| searches | The searches performed on the webset | List[Dict[str, Any]] |
| enrichments | The enrichments applied to the webset | List[Dict[str, Any]] |
| monitors | The monitors for the webset | List[Dict[str, Any]] |
| metadata | Key-value pairs associated with the webset | Dict[str, Any] |
| created_at | The date and time the webset was created | str |
| updated_at | The date and time the webset was last updated | str |

### Possible use case
<!-- MANUAL: use_case -->
**Configuration Review**: Retrieve webset details to verify settings before making changes.

**Status Checking**: Check the current status and configuration of a webset.

**Workflow Integration**: Fetch webset information for use in downstream workflow steps.
<!-- END MANUAL -->

---

## Exa List Websets

### What it is
List all Websets with pagination support

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves a paginated list of all your websets. Results include basic webset information and can be paginated through using cursor tokens.

Use this to discover existing websets, find specific websets by browsing, or build management interfaces for your webset collections.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| trigger | Trigger for the webset, value is ignored! | Any | No |
| cursor | Cursor for pagination through results | str | No |
| limit | Number of websets to return (1-100) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| websets | List of websets | List[Webset] |
| has_more | Whether there are more results to paginate through | bool |
| next_cursor | Cursor for the next page of results | str |

### Possible use case
<!-- MANUAL: use_case -->
**Inventory Management**: List all websets to understand your current data collections.

**Bulk Operations**: Iterate through websets to perform batch updates or cleanup.

**Dashboard Building**: Retrieve webset listings for management dashboards or reporting.
<!-- END MANUAL -->

---

## Exa Preview Webset

### What it is
Preview how a search query will be interpreted before creating a webset. Helps understand entity detection, criteria generation, and available enrichments.

### How it works
<!-- MANUAL: how_it_works -->
This block analyzes your search query and shows how Exa will interpret it before you create a webset. It reveals the detected entity type, generated criteria, and available enrichment columns.

Use this to refine your query and understand what results to expect. The block also provides suggestions for improving your query for better results.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Your search query to preview. Use this to see how Exa will interpret your search before creating a webset. | str | Yes |
| entity_type | Entity type to force: 'company', 'person', 'article', 'research_paper', or 'custom'. If not provided, Exa will auto-detect. | "company" \| "person" \| "article" \| "research_paper" \| "custom" \| "auto" | No |
| entity_description | Description for custom entity type (required when entity_type is 'custom') | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| preview | Full preview response with search and enrichment details | PreviewWebsetModel |
| entity_type | The detected or specified entity type | str |
| entity_description | Description of the entity type | str |
| criteria | Generated search criteria that will be used | List[PreviewCriterionModel] |
| enrichment_columns | Available enrichment columns that can be extracted | List[PreviewEnrichmentModel] |
| interpretation | Human-readable interpretation of how the query will be processed | str |
| suggestions | Suggestions for improving the query | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Query Optimization**: Test and refine search queries before committing to webset creation.

**Entity Validation**: Verify that Exa correctly detects the entity type for your use case.

**Enrichment Planning**: Discover available enrichment columns to plan data extraction.
<!-- END MANUAL -->

---

## Exa Update Webset

### What it is
Update metadata for an existing Webset

### How it works
<!-- MANUAL: how_it_works -->
This block updates the metadata associated with an existing webset. Metadata is stored as key-value pairs and can be used to organize, tag, or annotate websets.

Setting metadata to null clears all existing metadata. This operation does not affect the webset's items, searches, or enrichments.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to update | str | Yes |
| metadata | Key-value pairs to associate with this webset (set to null to clear) | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webset_id | The unique identifier for the webset | str |
| status | The status of the webset | str |
| external_id | The external identifier for the webset | str |
| metadata | Updated metadata for the webset | Dict[str, Any] |
| updated_at | The date and time the webset was updated | str |

### Possible use case
<!-- MANUAL: use_case -->
**Tagging Systems**: Add tags or labels to websets for organization and filtering.

**Project Association**: Link websets to specific projects or campaigns via metadata.

**Workflow State**: Store workflow-related state or flags in webset metadata.
<!-- END MANUAL -->

---

## Exa Webset Ready Check

### What it is
Check if webset is ready for next operation - enables conditional workflow branching

### How it works
<!-- MANUAL: how_it_works -->
This block checks if a webset is idle (no running operations) and optionally has a minimum number of items. It returns a boolean ready status along with a recommendation for the next action.

Use this block for conditional workflow branching to decide whether to proceed with processing, wait for more results, or add additional searches.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to check | str | Yes |
| min_items | Minimum number of items required to be 'ready' | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| is_ready | True if webset is idle AND has minimum items | bool |
| status | Current webset status | str |
| item_count | Number of items in webset | int |
| has_searches | Whether webset has any searches configured | bool |
| has_enrichments | Whether webset has any enrichments | bool |
| recommendation | Suggested next action (ready_to_process, waiting_for_results, needs_search, etc.) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Workflow Gating**: Only proceed with data processing when webset has enough items.

**Conditional Branching**: Route workflow based on webset readiness for different scenarios.

**Polling Logic**: Implement smart polling by checking readiness before fetching items.
<!-- END MANUAL -->

---

## Exa Webset Status

### What it is
Get a quick status overview of a webset

### How it works
<!-- MANUAL: how_it_works -->
This block returns a lightweight status overview of a webset without retrieving full item data. It includes counts for items, searches, enrichments, and monitors along with the current processing status.

Use this for quick status checks and monitoring without the overhead of retrieving complete webset details or items.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webset_id | The webset identifier | str |
| status | Current status (idle, running, paused, etc.) | str |
| item_count | Total number of items in the webset | int |
| search_count | Number of searches performed | int |
| enrichment_count | Number of enrichments configured | int |
| monitor_count | Number of monitors configured | int |
| last_updated | When the webset was last updated | str |
| is_processing | Whether any operations are currently running | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Status Dashboards**: Display webset status in monitoring dashboards.

**Health Checks**: Verify websets are active and processing as expected.

**Lightweight Polling**: Check status without fetching full webset data.
<!-- END MANUAL -->

---

## Exa Webset Summary

### What it is
Get a comprehensive summary of a webset with samples and statistics

### How it works
<!-- MANUAL: how_it_works -->
This block generates a comprehensive summary of a webset including statistics, sample items, and detailed breakdowns of searches and enrichments. It provides an overview useful for reporting and analysis.

You can control what to include in the summary such as sample items, search details, and enrichment details to balance comprehensiveness with response size.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| include_sample_items | Include sample items in the summary | bool | No |
| sample_size | Number of sample items to include | int | No |
| include_search_details | Include details about searches | bool | No |
| include_enrichment_details | Include details about enrichments | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webset_id | The webset identifier | str |
| status | Current status | str |
| entity_type | Type of entities in the webset | str |
| total_items | Total number of items | int |
| sample_items | Sample items from the webset | List[Dict[str, Any]] |
| search_summary | Summary of searches performed | SearchSummaryModel |
| enrichment_summary | Summary of enrichments applied | EnrichmentSummaryModel |
| monitor_summary | Summary of monitors configured | MonitorSummaryModel |
| statistics | Various statistics about the webset | WebsetStatisticsModel |
| created_at | When the webset was created | str |
| updated_at | When the webset was last updated | str |

### Possible use case
<!-- MANUAL: use_case -->
**Executive Reporting**: Generate summaries of webset collections for stakeholder reports.

**Quality Review**: Review sample items and statistics to assess webset quality.

**Progress Tracking**: Monitor webset growth and activity through periodic summaries.
<!-- END MANUAL -->

---
