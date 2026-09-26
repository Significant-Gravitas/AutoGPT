# Conductor Send Message
<!-- MANUAL: file_description -->
Sends a prompt to a Conductor agent session and, by default, waits for the agent to finish and returns its reply. Needs a `CONDUCTOR_API_KEY` credential, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys). Every Conductor block uses the same key.
<!-- END MANUAL -->

## Conductor Send Message

### What it is
Send a prompt to a Conductor agent session and, by default, wait for the agent to finish and return its reply.

### How it works
<!-- MANUAL: how_it_works -->
The block posts `{message}` to `POST /v0/sessions/{id}/messages` and returns the receipt (`message_id`, `state` queued or sent, `deep_link`). With `wait_for_reply` (the default) it then polls `GET /v0/sessions/{id}/status` every `poll_interval_seconds` until the agent is idle or errored, or `timeout_seconds` elapses (`timed_out` is set and whatever has arrived so far is returned), then lists the messages after the prompt. `reply` joins the text of the agent's messages; `messages` has the raw transcript entries. The block's own execution cap is two hours, so `timeout_seconds` is limited to that.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| session_id | Session to send the prompt to | str | Yes |
| message | Prompt for the agent | str | Yes |
| wait_for_reply | Wait until the agent is idle and return its reply | bool | No |
| timeout_seconds | How long to wait for the reply | int | No |
| poll_interval_seconds | Seconds between status checks while waiting | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message_id | ID of the sent prompt | str |
| state | queued or sent | str |
| deep_link | Link that opens the message | str |
| session_status | idle, working or error once waiting finished | str |
| reply | Text the agent produced in response | str |
| messages | Raw transcript messages after the prompt | List[Dict[str, Any]] |
| timed_out | True when the wait ended before the agent went idle | bool |
| error_message | Session error, if any | str |

### Possible use case
<!-- MANUAL: use_case -->
**Conversational driver**: Let AutoPilot steer a coding agent turn by turn: send a prompt, read `reply`, decide the next instruction.

**Fire and forget**: Turn `wait_for_reply` off to queue several prompts and check back later with Get Session.
<!-- END MANUAL -->

---
