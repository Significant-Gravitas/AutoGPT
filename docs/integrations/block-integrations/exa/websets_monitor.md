# Exa Websets Monitor
<!-- MANUAL: file_description -->
Blocks for creating scheduled monitors to automatically update Exa websets.
<!-- END MANUAL -->

## Exa Create Monitor

### What it is
Create automated monitors to keep websets updated with fresh data on a schedule

### How it works
<!-- MANUAL: how_it_works -->
This block creates a monitor (`POST /v0/monitors`) that runs a search on the webset on a schedule: a 5-field cron expression read in an IANA timezone (default Etc/UTC), at most one run per day. Each run looks for up to `search_count` (1-100) new items; a query, criteria or entity left empty reuses the webset's last search. 'append' adds the new items, while 'override' re-evaluates existing items against the run's criteria.

Exa removed refresh monitors, so the legacy 'refresh' behavior fails before calling Exa and the refresh content and enrichment inputs are ignored. The usual search checks apply: at most 5 criteria, and a 2-200 character description for a 'custom' entity. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | The ID of the Webset to monitor | str | Yes |
| cron_expression | When to run, as a 5-field Unix cron expression. Exa allows at most one run per day. | str | Yes |
| timezone | IANA timezone the cron expression is read in | str | No |
| behavior_type | What each run does. 'search' finds new items; 'refresh' is legacy (Exa removed it) and fails with an error. | "search" \| "refresh" | No |
| search_query | Query each run searches for (2-10,000 characters). Leave empty to reuse the webset's last search query. | str | No |
| search_count | Maximum number of new items to find per run | int | No |
| search_criteria | Up to 5 criteria new items must meet. Leave empty to reuse the last search's criteria. | List[str] | No |
| search_behavior | 'append' adds new items to the webset; 'override' re-evaluates existing items against the run's criteria | "append" \| "override" | No |
| entity_type | Kind of item to find: company, person, article, research_paper or custom. Leave empty to reuse the last search's entity. | str | No |
| entity_description | What each item is, when entity type is 'custom' (2-200 characters) | str | No |
| refresh_content | Legacy: only used by refresh monitors, which Exa removed | bool | No |
| refresh_enrichments | Legacy: only used by refresh monitors, which Exa removed | bool | No |
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

**Growing Datasets**: Schedule weekly searches that keep adding new matches to a research webset.
<!-- END MANUAL -->

---

## Exa Delete Monitor

### What it is
Delete a monitor from a webset

### How it works
<!-- MANUAL: how_it_works -->
This block permanently deletes a monitor, stopping all future scheduled runs. Any data already collected by the monitor remains in the webset.

Use this to clean up monitors that are no longer needed or to stop scheduled operations before deleting a webset. It charges only if Exa's response reports a cost (100 credits per USD).
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
This block retrieves a monitor and returns its status (enabled or disabled), behavior type and search config, cron schedule and timezone, next run time, details of the last run, metadata and timestamps.

Use this to verify monitor settings, check when the next run is scheduled, or review the most recent run. It charges only if Exa's response reports a cost (100 credits per USD).
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
| status | Current status: enabled or disabled | str |
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
This block returns one page of monitors (`limit` 1-200, default 25), optionally only those for one webset, both as a list and one at a time. Each entry includes the monitor's status, schedule, search config and next run time; pass `next_cursor` back in as `cursor` while `has_more` is true.

Use this to get an overview of all monitors or find the ones attached to a specific webset. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| webset_id | Filter monitors by webset ID | str | No |
| limit | Number of monitors to return (1-200) | int | No |
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
Update a monitor's status, schedule, search settings or metadata

### How it works
<!-- MANUAL: how_it_works -->
This block sends a PATCH with only the inputs you set. Status 'enabled' or 'disabled' turns scheduled runs on or off; the legacy 'paused' is sent as 'disabled'. A new timezone must come with the cron expression, and changing any search setting (query, criteria, behavior or entity) needs `search_count` too, because Exa requires it whenever the search is updated; otherwise the block fails before calling Exa.

Search settings left empty reuse the webset's last search. It charges only if Exa's response reports a cost (100 credits per USD).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| monitor_id | The ID of the monitor to update | str | Yes |
| status | Turn the monitor on ('enabled') or off ('disabled'). 'paused' is legacy and is sent as 'disabled'. | "enabled" \| "disabled" \| "paused" | No |
| cron_expression | New 5-field cron schedule (at most once per day). Required when changing the timezone. | str | No |
| timezone | New IANA timezone for the schedule (set together with the cron expression) | str | No |
| search_query | New query for each run (2-10,000 characters). Changing any search setting requires 'search count'. | str | No |
| search_count | New maximum number of items per run. Required to change the monitor's search settings. | int | No |
| search_criteria | New criteria for each run (up to 5). Leave empty to keep reusing the last search's criteria. | List[str] | No |
| search_behavior | 'append' adds new items; 'override' re-evaluates existing items | "append" \| "override" | No |
| entity_type | New kind of item to find: company, person, article, research_paper or custom | str | No |
| entity_description | What each item is, when entity type is 'custom' (2-200 characters) | str | No |
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

**Pause/Resume**: Disable monitors during maintenance or quiet periods and enable them again later.

**Search Tuning**: Change the query or criteria a monitor uses as your needs evolve.
<!-- END MANUAL -->

---
