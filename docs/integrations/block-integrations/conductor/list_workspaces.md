# Conductor List Workspaces
<!-- MANUAL: file_description -->
Lists Conductor workspaces with optional project, state, name, repository, creator and activity filters. Needs a `CONDUCTOR_API_KEY` credential, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys). Every Conductor block uses the same key.
<!-- END MANUAL -->

## Conductor List Workspaces

### What it is
List Conductor workspaces, optionally filtered by project, state, name, repository, creator or activity date.

### How it works
<!-- MANUAL: how_it_works -->
Without `project_id` the block calls `GET /v0/workspaces` and passes every filter as query parameters (`state` may repeat). With `project_id` it calls `GET /v0/projects/{id}/workspaces`, which only paginates, so the block applies the same filters itself to each returned page: exact `state` and `creator`, case-insensitive `name` and `repo` substrings, `since` against `lastActivityAt`, and archived workspaces hidden unless `include_archived` is set or `archived` is among the requested states. A filtered page can therefore be empty while `has_more` is still true; keep paging with `next_offset`, which always advances by the raw page size. `since` must be an ISO-8601 date or timestamp and is validated before any request. Each workspace is also emitted one at a time on `workspace` for fan-out.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project_id | Only workspaces of this project (repository). Leave empty for all projects. | str | No |
| state | Only workspaces in these states, e.g. ready, sleeping | List["initializing" \| "ready" \| "sleeping" \| "archived" \| "deleted" \| "updating" \| "unstarted"] | No |
| name | Filter by workspace name | str | No |
| repo | Filter by repository URL | str | No |
| creator | Filter by creator user ID | str | No |
| since | Only workspaces whose last activity is on or after this ISO-8601 date or timestamp | str | No |
| include_archived | Include archived workspaces | bool | No |
| limit | Maximum number of workspaces | int | No |
| offset | Pagination offset | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| workspaces | Workspaces: id, projectId, name, state, repoUrl, deepLink, creatorName, lastActivityAt | List[Dict[str, Any]] |
| workspace | Each workspace, one at a time | Dict[str, Any] |
| has_more | Whether more pages exist | bool |
| next_offset | Offset to request the next page; with project_id a page can be filtered down to nothing while has_more is still true | int |

### Possible use case
<!-- MANUAL: use_case -->
**Find work in progress**: Filter `state` to `ready` to see which workspaces have a live sandbox before sending prompts.

**Clean-up sweep**: List sleeping workspaces older than a `since` date and feed each `workspace` into Manage Workspace to archive it.
<!-- END MANUAL -->

---
