# Exa Websets Polling
<!-- MANUAL: file_description -->
Blocks for polling and waiting on Exa webset operations to complete.
<!-- END MANUAL -->

## Exa Wait For Enrichment

### What it is
Wait for a webset enrichment to complete with progress tracking

### How it works
<!-- MANUAL: how_it_works -->
This block polls the enrichment, starting every `check_interval` seconds (default 5) and backing off by 1.5x up to 30, until it is completed, failed or canceled, or `timeout` (default 300, max 1800) passes. On timeout it returns the last status with `timed_out` true instead of failing.

`items_enriched` and `sample_data` come from the webset's first 5 items only, and only when `sample_results` is on and the enrichment completed; otherwise `items_enriched` is 0. It charges only if Exa's response reports a cost (100 credits per USD).
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
This block polls the search, starting every `check_interval` seconds (default 5) and backing off by 1.5x up to 30, until it is completed, failed or canceled, or `timeout` (default 300, max 1800) passes. It returns items found, items analyzed, completion percentage and, once the search has finished, the recall estimate.

On timeout it returns the last known progress with `timed_out` true instead of failing. It charges only if Exa's response reports a cost (100 credits per USD).
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
This block polls a webset until it reaches the target status. For 'idle' (the default) and the legacy 'completed' and 'any_complete', it checks the webset every `check_interval` seconds until it is idle; if it is not idle within `timeout` (default 300, max 1800), the run fails with a timeout error. For 'running' or 'paused' it backs off by 1.5x up to `max_interval` and, on timeout, returns the last status with `timed_out` true.

It returns the final status, elapsed time, the number of items found by the webset's searches and, with `include_progress` on, per-search progress and per-enrichment status. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to monitor | str | Yes |
| target_status | Status to wait for: idle (all operations complete), running (actively processing) or paused. 'completed' and 'any_complete' are legacy and wait for idle. | "idle" \| "completed" \| "running" \| "paused" \| "any_complete" | No |
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
