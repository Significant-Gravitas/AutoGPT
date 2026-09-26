# Conductor Get Workspace
<!-- MANUAL: file_description -->
Everything about one Conductor workspace: details, status, shared preview URL and its agent sessions. Needs a `CONDUCTOR_API_KEY` credential, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys). Every Conductor block uses the same key.
<!-- END MANUAL -->

## Conductor Get Workspace

### What it is
Get everything about one Conductor workspace: details, current status, shared preview URL and its agent sessions.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `GET /v0/workspaces/{id}` and `GET /v0/workspaces/{id}/status`, then best-effort `GET /v0/workspaces/{id}/preview` and `GET /v0/workspaces/{id}/sessions` (a workspace that is still initializing may not serve those yet, in which case `preview_url` is empty and `sessions` is empty). `status` is one of initializing, ready, sleeping, archived, deleted, updating or unstarted; `lifecycle_step` and `error_message` explain an initializing or failed workspace.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| workspace_id | Workspace ID | str | Yes |
| include_archived_sessions | Include archived sessions in the session list | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| workspace | Workspace: id, projectId, name, state, repoUrl, deepLink, creatorName, lastActivityAt | Dict[str, Any] |
| status | initializing, ready, sleeping, archived, deleted, updating or unstarted | str |
| lifecycle_step | Setup step while initializing: building_snapshot, preparing, setting_up or updating | str |
| error_message | Workspace error, if any | str |
| preview_url | Public preview URL when a port is shared, else empty | str |
| preview_port | Port being shared at the preview URL, 0 when none | int |
| sessions | Agent sessions in the workspace: id, name, model, effort, deepLink | List[Dict[str, Any]] |
| deep_link | Link that opens the workspace | str |

### Possible use case
<!-- MANUAL: use_case -->
**Wait for readiness**: Poll this block after Create Workspace until `status` is `ready` before opening a second session.

**Grab the preview link**: Read `preview_url` after Manage Workspace shared a port, and post it to Slack for review.

**Pick a session**: Use `sessions[].id` to choose which chat to send a follow-up prompt to.
<!-- END MANUAL -->

---
