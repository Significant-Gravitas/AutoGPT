# Exa Create Monitor

### What it is
Create automated monitors to keep websets updated with fresh data on a schedule.

### What it does
Create automated monitors to keep websets updated with fresh data on a schedule

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to monitor | str | Yes |
| cron_expression | Cron expression for scheduling (5 fields, max once per day) | str | Yes |
| timezone | IANA timezone for the schedule | str | No |
| behavior_type | Type of monitor behavior (search for new items or refresh existing) | "search" | "refresh" | No |
| search_query | Search query for finding new items (required for search behavior) | str | No |
| search_count | Number of items to find in each search | int | No |
| search_criteria | Criteria that items must meet | List[str] | No |
| search_behavior | How new results interact with existing items | "append" | "override" | No |
| entity_type | Type of entity to search for (company, person, etc.) | str | No |
| refresh_content | Refresh content from source URLs (for refresh behavior) | bool | No |
| refresh_enrichments | Re-run enrichments on items (for refresh behavior) | bool | No |
| metadata | Metadata to attach to the monitor | Dict[str, True] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| monitor_id | The unique identifier for the created monitor | str |
| webset_id | The webset this monitor belongs to | str |
| status | Status of the monitor | str |
| behavior_type | Type of monitor behavior | str |
| next_run_at | When the monitor will next run | str |
| cron_expression | The schedule cron expression | str |
| timezone | The timezone for scheduling | str |
| created_at | When the monitor was created | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Delete Monitor

### What it is
Delete a monitor from a webset.

### What it does
Delete a monitor from a webset

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| monitor_id | The ID of the monitor to delete | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| monitor_id | The ID of the deleted monitor | str |
| success | Whether the deletion was successful | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Get Monitor

### What it is
Get the details and status of a webset monitor.

### What it does
Get the details and status of a webset monitor

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| monitor_id | The ID of the monitor to retrieve | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| monitor_id | The unique identifier for the monitor | str |
| webset_id | The webset this monitor belongs to | str |
| status | Current status of the monitor | str |
| behavior_type | Type of monitor behavior | str |
| behavior_config | Configuration for the monitor behavior | Dict[str, True] |
| cron_expression | The schedule cron expression | str |
| timezone | The timezone for scheduling | str |
| next_run_at | When the monitor will next run | str |
| last_run | Information about the last run | Dict[str, True] |
| created_at | When the monitor was created | str |
| updated_at | When the monitor was last updated | str |
| metadata | Metadata attached to the monitor | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa List Monitors

### What it is
List all monitors with optional webset filtering.

### What it does
List all monitors with optional webset filtering

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | Filter monitors by webset ID | str | No |
| limit | Number of monitors to return | int | No |
| cursor | Cursor for pagination | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| monitors | List of monitors | List[Dict[str, True]] |
| monitor | Individual monitor (yielded for each monitor) | Dict[str, True] |
| has_more | Whether there are more monitors to paginate through | bool |
| next_cursor | Cursor for the next page of results | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Exa Update Monitor

### What it is
Update a monitor's status, schedule, or metadata.

### What it does
Update a monitor's status, schedule, or metadata

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| monitor_id | The ID of the monitor to update | str | Yes |
| status | New status for the monitor | "enabled" | "disabled" | "paused" | No |
| cron_expression | New cron expression for scheduling | str | No |
| timezone | New timezone for the schedule | str | No |
| metadata | New metadata for the monitor | Dict[str, True] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| monitor_id | The unique identifier for the monitor | str |
| status | Updated status of the monitor | str |
| next_run_at | When the monitor will next run | str |
| updated_at | When the monitor was updated | str |
| success | Whether the update was successful | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
