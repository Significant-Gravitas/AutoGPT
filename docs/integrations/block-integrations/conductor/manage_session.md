# Conductor Manage Session
<!-- MANUAL: file_description -->
Renames, cancels or archives a Conductor agent session. Needs a `CONDUCTOR_API_KEY` credential, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys). Every Conductor block uses the same key.
<!-- END MANUAL -->

## Conductor Manage Session

### What it is
Rename, cancel or archive a Conductor agent session. Cancel stops the running turn and drops queued prompts.

### How it works
<!-- MANUAL: how_it_works -->
One `action` per run: `rename` posts `{name}` to `/rename`; `cancel` posts to `/cancel`, stopping the current turn and dropping queued prompts; `archive` posts to `/archive`. Cancel and archive report `canceled_queued_messages`, and `status` reflects the session state afterwards when Conductor returns it.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| session_id | Session ID | str | Yes |
| action | rename, cancel (stop the current turn and drop queued prompts) or archive | "rename" \| "cancel" \| "archive" | No |
| name | New session name (rename) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| session_id | Session ID | str |
| workspace_id | Workspace the session belongs to, when reported | str |
| status | Session status after the action, when reported | str |
| canceled_queued_messages | Queued prompts dropped by cancel or archive | int |
| result | Raw API response | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Stop a runaway agent**: Cancel a session whose `status` has been `working` past a deadline.

**Housekeeping**: Archive finished sessions so Get Workspace lists only active chats.
<!-- END MANUAL -->

---
