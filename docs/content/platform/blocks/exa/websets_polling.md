# Exa Wait For Enrichment

### What it is
Wait for a webset enrichment to complete with progress tracking.

### What it does
Wait for a webset enrichment to complete with progress tracking

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Wait For Search

### What it is
Wait for a specific webset search to complete with progress tracking.

### What it does
Wait for a specific webset search to complete with progress tracking

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
| recall_info | Information about expected results and confidence | Dict[str, True] |
| timed_out | Whether the operation timed out | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Wait For Webset

### What it is
Wait for a webset to reach a specific status with progress tracking.

### What it does
Wait for a webset to reach a specific status with progress tracking

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to monitor | str | Yes |
| target_status | Status to wait for (idle=all operations complete, completed=search done, running=actively processing) | "idle" | "completed" | "running" | No |
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
| search_progress | Detailed search progress information | Dict[str, True] |
| enrichment_progress | Detailed enrichment progress information | Dict[str, True] |
| timed_out | Whether the operation timed out | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
