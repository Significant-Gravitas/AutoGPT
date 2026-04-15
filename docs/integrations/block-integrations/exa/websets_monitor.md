# Exa Websets Monitor
<!-- MANUAL: file_description -->
Blocks for creating scheduled monitors to automatically update Exa websets.
<!-- END MANUAL -->

## Exa Create Monitor

### What it is
Create automated monitors to keep websets updated with fresh data on a schedule

### How it works
<!-- MANUAL: how_it_works -->
This block creates a scheduled monitor that automatically updates a webset on a cron schedule. Monitors can either search for new items matching criteria or refresh existing item content and enrichments.

Configure the cron expression for your desired frequency (daily, weekly, etc.) and choose between search behavior to find new items or refresh behavior to update existing data.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID or external ID of the Webset to monitor | str | Yes |
| cron_expression | Cron expression for scheduling (5 fields, max once per day) | str | Yes |
| timezone | IANA timezone for the schedule | str | No |
| behavior_type | Type of monitor behavior (search for new items or refresh existing) | "search" \| "refresh" | No |
| search_query | Search query for finding new items (required for search behavior) | str | No |
| search_count | Number of items to find in each search | int | No |
| search_criteria | Criteria that items must meet | List[str] | No |
| search_behavior | How new results interact with existing items | "append" \| "override" | No |
| entity_type | Type of entity to search for (company, person, etc.) | str | No |
| refresh_content | Refresh content from source URLs (for refresh behavior) | bool | No |
| refresh_enrichments | Re-run enrichments on items (for refresh behavior) | bool | No |
| metadata | Metadata to attach to the monitor | Dict[str, Any] | No |

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
**Continuous Lead Generation**: Schedule daily searches to find new companies matching your criteria.

**News Monitoring**: Set up monitors to discover fresh articles on topics of interest.

**Data Freshness**: Schedule periodic refreshes to keep enrichment data current.
<!-- END MANUAL -->

---

## Exa Delete Monitor

### What it is
Delete a monitor from a webset

### How it works
<!-- MANUAL: how_it_works -->
This block permanently deletes a monitor, stopping all future scheduled runs. Any data already collected by the monitor remains in the webset.

Use this to clean up monitors that are no longer needed or to stop scheduled operations before deleting a webset.
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
**Project Completion**: Delete monitors when monitoring campaigns or projects conclude.

**Cost Management**: Remove monitors that are no longer providing value to reduce costs.

**Configuration Cleanup**: Delete old monitors before creating updated replacements.
<!-- END MANUAL -->

---

## Exa Get Monitor

### What it is
Get the details and status of a webset monitor

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves detailed information about a monitor including its configuration, schedule, current status, and information about the last run.

Use this to verify monitor settings, check when the next run is scheduled, or review results from recent executions.
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
| behavior_config | Configuration for the monitor behavior | Dict[str, Any] |
| cron_expression | The schedule cron expression | str |
| timezone | The timezone for scheduling | str |
| next_run_at | When the monitor will next run | str |
| last_run | Information about the last run | Dict[str, Any] |
| created_at | When the monitor was created | str |
| updated_at | When the monitor was last updated | str |
| metadata | Metadata attached to the monitor | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Schedule Verification**: Check when monitors are scheduled to run next.

**Performance Review**: Examine last run details to assess monitor effectiveness.

**Configuration Audit**: Retrieve monitor settings for documentation or troubleshooting.
<!-- END MANUAL -->

---

## Exa List Monitors

### What it is
List all monitors with optional webset filtering

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves a paginated list of all monitors, optionally filtered by webset. Results include basic monitor information such as status, schedule, and next run time.

Use this to get an overview of all active monitors or find monitors associated with a specific webset.
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
| monitors | List of monitors | List[Dict[str, Any]] |
| monitor | Individual monitor (yielded for each monitor) | Dict[str, Any] |
| has_more | Whether there are more monitors to paginate through | bool |
| next_cursor | Cursor for the next page of results | str |

### Possible use case
<!-- MANUAL: use_case -->
**Monitor Dashboard**: Build dashboards showing all active monitors and their schedules.

**Webset Management**: Find monitors associated with websets before making changes.

**Activity Overview**: Review all scheduled monitoring activity across your account.
<!-- END MANUAL -->

---

## Exa Update Monitor

### What it is
Update a monitor's status, schedule, or metadata

### How it works
<!-- MANUAL: how_it_works -->
This block modifies an existing monitor's configuration. You can enable, disable, or pause monitors, change their schedule, update the timezone, or modify metadata.

Changes take effect immediately. Disabling a monitor stops future scheduled runs until re-enabled.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| monitor_id | The ID of the monitor to update | str | Yes |
| status | New status for the monitor | "enabled" \| "disabled" \| "paused" | No |
| cron_expression | New cron expression for scheduling | str | No |
| timezone | New timezone for the schedule | str | No |
| metadata | New metadata for the monitor | Dict[str, Any] | No |

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
**Schedule Changes**: Adjust monitor frequency based on data velocity or business needs.

**Pause/Resume**: Temporarily pause monitors during maintenance or when not needed.

**Status Management**: Enable or disable monitors programmatically based on conditions.
<!-- END MANUAL -->

---
