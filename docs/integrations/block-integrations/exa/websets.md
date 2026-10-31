# Exa Websets
<!-- MANUAL: file_description -->
Blocks for creating, finding, inspecting and managing Exa websets, persistent collections of web entities that Exa finds and checks against your criteria.
<!-- END MANUAL -->

## Exa Cancel Webset

### What it is
Cancel all operations being performed on a Webset

### How it works
<!-- MANUAL: how_it_works -->
This block sends `POST /v0/websets/{id}/cancel` to the Websets API, which stops every running operation (searches, enrichments) on the webset. It accepts the webset ID or your external ID and returns the status Exa reports after cancellation. Items already processed before cancellation are retained.

The block is useful for stopping long-running operations that are no longer needed or when you need to modify the webset configuration. It charges only if Exa's response reports a cost (100 credits per USD).
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
This block first looks the webset up by your external ID (`GET /v0/websets/{external_id}`). If it exists, the block returns it unchanged with `was_created` false, even when the search query or metadata you pass are different. Only a 404 leads to creation; any other error fails the run.

When creating, the block sends the external ID, the metadata and, if you give a search query, a search for `search_count` items (1-1000). It returns as soon as Exa accepts the webset and does not wait for the search to finish. It charges only if Exa's response reports a cost (100 credits per USD).
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
This block sends one `POST /v0/websets` request that starts a search and can also copy in items from imports or other websets, add enrichment columns, and set webset-wide exclusions, a title, an external ID and metadata. The search can be limited to imports or websets (scope), leave out the items of others (exclude), or become a "hop" search when a scope source has a relationship such as "investors of these companies", which needs a limit of 1-10. Before calling Exa the block rejects more than 5 criteria, a 'custom' entity without a 2-200 character description, a source type other than 'import' or 'webset', and more than 150 options on an enrichment.

With `wait_for_initial_results` on (the default), the block polls the webset every 5 seconds until it is idle, then returns it with the number of items its searches found and the time taken. If the webset is still busy after `polling_timeout` seconds (default 300, max 600), the run fails with a timeout error, although the webset has already been created. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| search_query | What to find, in plain language (up to 5,000 characters). Be specific about the characteristics every result must have. Any URL provided will be crawled and used as context for the search. | str | Yes |
| title | Name shown wherever the webset is displayed. Leave empty to let Exa generate one. | str | No |
| search_count | Number of items the search will attempt to find. The actual number of items found may be less than this number depending on the search complexity. | int | No |
| search_entity_type | Kind of result to find: company, person, article, research_paper, or custom (describe it below). 'auto' lets Exa detect it from the query. | "company" \| "person" \| "article" \| "research_paper" \| "custom" \| "auto" | No |
| search_entity_description | What each result is, when the entity type is 'custom' (2-200 characters, required for custom). | str | No |
| search_criteria | Up to 5 criteria every item must satisfy (each up to 1,000 characters). Leave empty to have Exa generate them from the query. | List[str] | No |
| search_recall | Also estimate how many matching results exist on the web. The estimate appears in the search's 'recall' field. | bool | No |
| search_max_people_per_company | For people searches: try to include at most this many people from the same current employer (soft cap). | int | No |
| search_exclude_sources | IDs of imports or websets whose items this search must leave out | List[str] | No |
| search_exclude_types | Source type for each ID above, in the same order ('import' or 'webset'; defaults to 'import') | List["import" \| "webset"] | No |
| search_scope_sources | IDs of imports or websets to search within instead of the whole web | List[str] | No |
| search_scope_types | Source type for each scope ID, in the same order ('import' or 'webset'; defaults to 'import') | List["import" \| "webset"] | No |
| search_scope_relationships | Optional, one per scope source: how the results relate to that source's items, for hop searches (e.g. 'investors of these companies') | List[str] | No |
| search_scope_relationship_limits | Required for each relationship above: how many related entities to find per source item (1-10) | List[int] | No |
| import_sources | IDs of imports or websets whose items are copied into this webset | List[str] | No |
| import_types | Source type for each import ID, in the same order ('import' or 'webset'; defaults to 'import') | List["import" \| "webset"] | No |
| import_evaluate | Optional, one per import source: check that source's items against the search criteria (requires a search) | List[bool] | No |
| exclude_sources | IDs of imports or websets whose items are left out of every search and import in this webset | List[str] | No |
| exclude_types | Source type for each webset-wide exclude ID, in the same order ('import' or 'webset'; defaults to 'import') | List["import" \| "webset"] | No |
| enrichment_descriptions | Data to extract for every item, one description per enrichment column (up to 5,000 characters each) | List[str] | No |
| enrichment_formats | Format for each enrichment, in the same order: text, date, number, options, email, phone or url. If not specified, Exa picks the best format. | List["text" \| "date" \| "number" \| "options" \| "email" \| "phone" \| "url"] | No |
| enrichment_options | For 'options' enrichments, in the same order: the labels to choose from (1-150 per enrichment) | List[List[str]] | No |
| enrichment_metadata | Optional metadata for each enrichment, in the same order | List[Dict[str, Any]] | No |
| external_id | Your own identifier for the webset (up to 300 characters). Other Websets blocks accept it in place of the webset ID. | str | No |
| metadata | Key-value pairs to store on the webset (values up to 1,000 characters) | Dict[str, Any] | No |
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

Use this to clean up websets that are no longer needed or to remove test data. The block accepts either the Exa-generated ID or your custom external_id. It charges only if Exa's response reports a cost (100 credits per USD).
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

**Stop Scheduled Work**: Delete a webset you no longer need so its monitors stop running scheduled searches.
<!-- END MANUAL -->

---

## Exa Get Webset

### What it is
Retrieve a Webset by ID or external ID

### How it works
<!-- MANUAL: how_it_works -->
This block sends `GET /v0/websets/{id}` (webset ID or external ID) and returns the webset's status, title, dashboard link, metadata and timestamps, plus its searches, imports, enrichments, monitors and webset-wide excludes as raw objects. Turn on `expand_items` to also get up to the first 100 items in `items`; otherwise `items` is empty.

Use this to check webset configuration or status before performing operations. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to retrieve | str | Yes |
| expand_items | Also return the webset's first 100 items in the 'items' output | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webset_id | The unique identifier for the webset | str |
| status | The status of the webset: idle, pending, running or paused | str |
| external_id | The external identifier for the webset | str |
| title | The webset's name | str |
| dashboard_url | Link to the webset in the Exa dashboard | str |
| searches | The searches performed on the webset | List[Dict[str, Any]] |
| imports | The imports that brought items into the webset | List[Dict[str, Any]] |
| enrichments | The enrichments applied to the webset | List[Dict[str, Any]] |
| monitors | The monitors for the webset | List[Dict[str, Any]] |
| excludes | Imports or websets excluded from every operation in this webset | List[Dict[str, Any]] |
| items | Up to the first 100 items (empty unless 'expand items' is on) | List[Dict[str, Any]] |
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
List all Websets with pagination support, optionally filtered by ID, external ID or title

### How it works
<!-- MANUAL: how_it_works -->
This block sends `GET /v0/websets` and returns one page of websets (`limit` 1-100, default 25), both as a list and one at a time. `search` keeps only websets whose ID, external ID or title matches the text (2-50 characters). While `has_more` is true, pass `next_cursor` back in as `cursor` to get the next page. The `trigger` input's value is ignored; it only lets another block start this one.

Use this to discover existing websets or build management views of your collections. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| trigger | Trigger for the webset, value is ignored! | Any | No |
| search | Only return websets whose ID, external ID or title matches this text (2-50 characters) | str | No |
| cursor | Cursor for pagination through results | str | No |
| limit | Number of websets to return (1-100) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| websets | List of websets | List[Webset] |
| webset | Individual webset (yielded for each webset in the list) | Webset |
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
This block sends your query to `POST /v0/websets/preview` and shows how Exa will interpret it before you create a webset: the detected (or forced) entity type, the criteria Exa would generate, and the enrichment columns it detects. A 'custom' entity type needs a 2-200 character description. Turn on `include_sample_items` to also get up to `sample_count` (1-10) matching items; this runs a real search and uses Websets credits.

The `interpretation` sentence and `suggestions` are written by the block from the number of criteria and columns, not by Exa: it suggests adding criteria when none were generated and naming the data points you want when no columns were detected. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Your search query to preview. Use this to see how Exa will interpret your search before creating a webset. | str | Yes |
| entity_type | Entity type to force: 'company', 'person', 'article', 'research_paper', or 'custom'. If not provided, Exa will auto-detect. | "company" \| "person" \| "article" \| "research_paper" \| "custom" \| "auto" | No |
| entity_description | Description for custom entity type (2-200 characters, required when entity_type is 'custom') | str | No |
| include_sample_items | Also search for a few matching items and return them in 'items'. This runs a real search and uses Websets credits. | bool | No |
| sample_count | How many sample items to return when 'include sample items' is on (1-10) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| preview | Full preview response with search and enrichment details | PreviewWebsetModel |
| entity_type | The detected or specified entity type | str |
| entity_description | Description of the entity type | str |
| criteria | Generated search criteria that will be used | List[PreviewCriterionModel] |
| enrichment_columns | Available enrichment columns that can be extracted | List[PreviewEnrichmentModel] |
| items | Sample matching items (empty unless 'include sample items' is on) | List[Dict[str, Any]] |
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
Update the title or metadata of an existing Webset

### How it works
<!-- MANUAL: how_it_works -->
This block sends `POST /v0/websets/{id}` (webset ID or external ID) with a new title and/or metadata. Inputs you leave empty are not sent, so the current title or metadata stays as it is; leaving metadata empty does not clear it.

This operation does not affect the webset's items, searches, or enrichments. It returns the webset's status, title, metadata and update time as Exa reports them after the change. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to update | str | Yes |
| title | New name for the webset. Leave empty to keep the current one. | str | No |
| metadata | Key-value pairs to associate with this webset (set to null to clear) | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webset_id | The unique identifier for the webset | str |
| status | The status of the webset | str |
| external_id | The external identifier for the webset | str |
| title | The webset's name | str |
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
This block reads the webset once and sets `is_ready` to true when its status is idle and it has at least `min_items` items (default 1). The item count is the total found by the webset's searches, because Exa reports no webset-wide total, so items that came only from imports are not counted. `recommendation` is `needs_search` (no searches), `waiting_for_results` (running or pending), `insufficient_items`, `ready_to_enrich` (no enrichments yet) or `ready_to_process`.

Use this block for conditional workflow branching to decide whether to proceed with processing, wait for more results, or add additional searches. It charges only if Exa's response reports a cost (100 credits per USD).
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
This block reads the webset once and returns its status (idle, pending, running or paused), the number of searches, enrichments and monitors, and when it was last updated, without fetching any items. `item_count` is the total found by the webset's searches (Exa reports no webset-wide total, so imported items are not included), and `is_processing` is true while the webset is running or pending.

Use this for quick status checks and monitoring without the overhead of retrieving complete webset details or items. It charges only if Exa's response reports a cost (100 credits per USD).
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
| status | Current status (idle, pending, running or paused) | str |
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
This block reads the webset and, when `include_sample_items` is on, fetches its first `sample_size` items (0-10, default 3). It returns summaries of searches (counts, items found, first three queries), enrichments (counts, formats, first three titles) and monitors (enabled count, earliest next run), plus statistics such as average items per search. Turning off `include_search_details` or `include_enrichment_details` sets that summary to zeros instead of leaving it out.

`total_items` is the number of sample items returned, not the size of the webset; use `search_summary.total_items_found` or the Webset Status block for the item count. It charges only if Exa's response reports a cost (100 credits per USD).
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
