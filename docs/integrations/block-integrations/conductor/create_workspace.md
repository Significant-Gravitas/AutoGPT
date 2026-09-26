# Conductor Create Workspace
<!-- MANUAL: file_description -->
Creates a Conductor cloud workspace for a project or repository, optionally starts its agent with a prompt and waits for the reply. Needs a `CONDUCTOR_API_KEY` credential, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys). Every Conductor block uses the same key.
<!-- END MANUAL -->

## Conductor Create Workspace

### What it is
Create a Conductor cloud workspace for a project or repository, optionally start its agent with a prompt and wait for the reply.

### How it works
<!-- MANUAL: how_it_works -->
The block posts to `POST /v0/workspaces` with exactly one of `project_id` or `repository_url` (both or neither is an input error). Blank optional fields are omitted so Conductor applies its defaults; `model` is passed through as-is, so use an id Conductor accepts (for example `fable-5-1`, `opus-5-5-1m`, `sonnet-5-1m`, `gpt-6-astra` or `auto`). When `message` is set the agent starts on it immediately and `initial_message_id` is returned. With `wait_for_reply` the block then polls `GET /v0/sessions/{id}/status` every `poll_interval_seconds` until the agent is idle or errored (or `timeout_seconds` elapses, reported through `timed_out`) and returns the transcript messages after the prompt with their agent text joined into `reply`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| project_id | Project (repository) to open the workspace in. Find IDs with Get Account. Use this or repository_url, not both. | str | No |
| repository_url | Git repository URL to open instead of a project ID | str | No |
| message | Initial prompt for the agent. Leave empty to create an idle workspace. | str | No |
| branch | Branch to start from | str | No |
| name | Workspace name | str | No |
| session_name | Name of the initial agent session | str | No |
| agent | Agent for the initial session | "claude" \| "codex" \| "cursor" \| "acp" | No |
| model | Model id such as fable-5-1, opus-5-5-1m, sonnet-5-1m, gpt-6-astra or auto. Leave empty for Conductor's default. | str | No |
| effort | Reasoning effort; leave empty for the default | "" \| "none" \| "low" \| "medium" \| "high" \| "xhigh" \| "max" \| "ultra" | No |
| fast_mode | Enable fast mode | bool | No |
| env | Environment variables for the workspace | Dict[str, str] | No |
| restricted_access | Restrict the workspace to its creator | bool | No |
| wait_for_reply | After sending the initial prompt, wait until the agent is idle and return its reply | bool | No |
| timeout_seconds | How long to wait for the reply | int | No |
| poll_interval_seconds | Seconds between status checks while waiting | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| workspace_id | ID of the new workspace | str |
| session_id | ID of the initial session | str |
| deep_link | Link that opens the workspace | str |
| initial_message_id | ID of the initial prompt message, empty when none was sent | str |
| session_status | idle, working or error once waiting finished | str |
| reply | Text the agent produced in response | str |
| messages | Raw transcript messages after the prompt | List[Dict[str, Any]] |
| timed_out | True when the wait ended before the agent went idle | bool |
| error_message | Session error, if any | str |

### Possible use case
<!-- MANUAL: use_case -->
**Ticket to branch**: Take an issue description from a trigger, create a workspace on the repo with that prompt, and wait for the agent's summary of the change.

**Parallel exploration**: Fan one task out to several workspaces with different `model` or `agent` values and compare the replies.
<!-- END MANUAL -->

---
