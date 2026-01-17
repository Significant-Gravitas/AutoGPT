# Exa Websets Polling
<!-- MANUAL: file_description -->
Blocks for polling and waiting on Exa webset operations to complete.
<!-- END MANUAL -->

## Exa Wait For Enrichment

### What it is
Wait for a webset enrichment to complete with progress tracking

### How it works
<!-- MANUAL: how_it_works -->
This block polls an enrichment operation until it completes or times out. It checks status at configurable intervals and can include sample results when done.

Use this to block workflow execution until enrichments finish, enabling sequential operations that depend on enrichment data being available.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| enrichment_id | The ID of the enrichment to monitor | str | Yes |
| timeout | Maximum time to wait in seconds | int | No |
| check_interval | Initial interval between status checks in seconds | int | No |
| sample_results | Include sample enrichment results in output | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| enrichment_id | The enrichment ID that was monitored | str |
| final_status | The final status of the enrichment | str |
| items_enriched | Number of items successfully enriched | int |
| enrichment_title | Title/description of the enrichment | str |
| elapsed_time | Total time elapsed in seconds | float |
| sample_data | Sample of enriched data (if requested) | List[SampleEnrichmentModel] |
| timed_out | Whether the operation timed out | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Sequential Processing**: Wait for enrichments to complete before proceeding to export or analysis.

**Data Validation**: Ensure enrichments finish and review samples before continuing workflow.

**Synchronous Workflows**: Convert async enrichment operations to blocking calls for simpler logic.
<!-- END MANUAL -->

---

## Exa Wait For Search

### What it is
Wait for a specific webset search to complete with progress tracking

### How it works
<!-- MANUAL: how_it_works -->
This block polls a webset search operation until it completes or times out. It provides progress information including items found, items analyzed, and completion percentage.

Use this when you need search results before proceeding with downstream operations like enrichments or exports.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset | str | Yes |
| search_id | The ID of the search to monitor | str | Yes |
| timeout | Maximum time to wait in seconds | int | No |
| check_interval | Initial interval between status checks in seconds | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| search_id | The search ID that was monitored | str |
| final_status | The final status of the search | str |
| items_found | Number of items found by the search | int |
| items_analyzed | Number of items analyzed | int |
| completion_percentage | Completion percentage (0-100) | int |
| elapsed_time | Total time elapsed in seconds | float |
| recall_info | Information about expected results and confidence | Dict[str, Any] |
| timed_out | Whether the operation timed out | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Search Completion**: Wait for initial webset population before accessing items.

**Progress Monitoring**: Track search progress in long-running operations.

**Sequential Workflows**: Ensure searches complete before starting enrichments.
<!-- END MANUAL -->

---

## Exa Wait For Webset

### What it is
Wait for a webset to reach a specific status with progress tracking

### How it works
<!-- MANUAL: how_it_works -->
This block polls a webset until it reaches a target status (idle, completed, or running). It uses exponential backoff for efficient polling and provides detailed progress information.

Use this for general-purpose waiting on webset operations when you don't need to track a specific search or enrichment.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to monitor | str | Yes |
| target_status | Status to wait for (idle=all operations complete, completed=search done, running=actively processing) | "idle" \| "completed" \| "running" \| "paused" \| "any_complete" | No |
| timeout | Maximum time to wait in seconds | int | No |
| check_interval | Initial interval between status checks in seconds | int | No |
| max_interval | Maximum interval between checks (for exponential backoff) | int | No |
| include_progress | Include detailed progress information in output | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| webset_id | The webset ID that was monitored | str |
| final_status | The final status of the webset | str |
| elapsed_time | Total time elapsed in seconds | float |
| item_count | Number of items found | int |
| search_progress | Detailed search progress information | Dict[str, Any] |
| enrichment_progress | Detailed enrichment progress information | Dict[str, Any] |
| timed_out | Whether the operation timed out | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Workflow Orchestration**: Wait for all webset operations to complete before next workflow steps.

**Idle State Detection**: Ensure webset is fully idle before making configuration changes.

**Completion Gates**: Block workflow until webset reaches a specific readiness state.
<!-- END MANUAL -->

---
