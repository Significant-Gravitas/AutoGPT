# Conductor Create Session
<!-- MANUAL: file_description -->
Starts a new agent session (chat) in an existing Conductor workspace, optionally with a first prompt, and optionally waits for the reply. Needs a `CONDUCTOR_API_KEY` credential, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys). Every Conductor block uses the same key.
<!-- END MANUAL -->

## Conductor Create Session

### What it is
Start a new agent session (chat) in an existing Conductor workspace, optionally with a first prompt, and optionally wait for the agent's reply.

### How it works
<!-- MANUAL: how_it_works -->
The block posts to `POST /v0/sessions` with `workspaceId`, `agent` and any non-blank `model`, `effort`, `name` or `message`. When `message` is given the returned `initial_message_id` identifies the prompt; with `wait_for_reply` the block waits the same way as Send Message: it polls the session status (bounded by `timeout_seconds`), correlates the transcript rows with the prompt's turn, and returns them in `messages` with their visible agent text joined into `reply` (`truncated` is set when the turn exceeded the 1000 rows kept).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| workspace_id | Workspace to add the session to | str | Yes |
| message | Initial prompt for the agent; leave empty for an idle session | str | No |
| agent | Agent to run | "claude" \| "codex" \| "cursor" \| "acp" | No |
| model | Model id such as fable-5-1, opus-5-5-1m, sonnet-5-1m, gpt-6-astra or auto. Leave empty for Conductor's default. | str | No |
| effort | Reasoning effort; leave empty for the default | "" \| "none" \| "low" \| "medium" \| "high" \| "xhigh" \| "max" \| "ultra" | No |
| fast_mode | Enable fast mode | bool | No |
| name | Session name | str | No |
| wait_for_reply | After sending the initial prompt, wait until the agent is idle and return its reply | bool | No |
| timeout_seconds | How long to wait for the reply | int | No |
| poll_interval_seconds | Seconds between status checks while waiting | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| session_id | ID of the new session | str |
| deep_link | Link that opens the session | str |
| initial_message_id | ID of the initial prompt message, empty when none was sent | str |
| session_status | idle, working or error once waiting finished | str |
| reply | Text the agent produced in response | str |
| messages | Raw transcript messages after the prompt | List[Dict[str, Any]] |
| timed_out | True when the wait ended before the agent went idle | bool |
| truncated | True when the turn produced more messages than are kept; messages holds the newest ones and reply may be incomplete | bool |
| error_message | Session error, if any | str |

### Possible use case
<!-- MANUAL: use_case -->
**Second opinion**: Open a Codex session in a workspace where Claude already worked and ask it to review the diff.

**Split work**: Start one session per sub-task in the same workspace so they share the checkout.
<!-- END MANUAL -->

---
