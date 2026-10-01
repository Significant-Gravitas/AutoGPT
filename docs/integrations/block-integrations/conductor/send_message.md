# Conductor Send Message
<!-- MANUAL: file_description -->
Sends a prompt to a Conductor agent session and, by default, waits for the agent to finish and returns its reply. Requires your own Conductor API key, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys) and added through AutoGPT’s credentials UI. Select that credential for each Conductor block; no server-wide default key is used.
<!-- END MANUAL -->

## Conductor Send Message

### What it is
Send a prompt to a Conductor agent session and, by default, wait for the agent to finish and return its reply. Coding agents typically run 5-30 minutes: prefer a single wait with a large timeout_seconds (max 7200) over repeated short polls or scheduled follow-ups. When timed_out is true the agent is still working; keep waiting with Conductor Get Session (wait_until_idle=true, after=next_after, prompt_message_id=the prompt's message id) rather than scheduling a check.

### How it works
<!-- MANUAL: how_it_works -->
The block posts `{message}` to `POST /v0/sessions/{id}/messages` and returns the receipt (`message_id`, `state` queued or sent, `deep_link`). With `wait_for_reply` (the default) it then polls every `poll_interval_seconds`: each poll reads the transcript rows that arrived since the previous one and then `GET /v0/sessions/{id}/status`, so the status is never older than the rows it is judged against. The receipt ID is the prompt's `content.id` in the transcript (not a row ID, so it cannot be used as a cursor); the block looks for that row among the newest 300 rows, then in up to 1000 older rows, and otherwise resolves the turn from agent rows tagged with the receipt as their `turnId`, reporting the omitted history through `truncated`. It finishes when the session reports `error`, or `idle` after the turn has progressed past its startup events (Claude `system`/`command_lifecycle`, Codex `thread.started`/`turn.started`), reading the transcript once more for rows written just before the status changed; a session that is idle because the prompt is still queued, or has only launched the agent, is not treated as finished. `reply` joins the visible agent text of the turn (Claude `assistant` text, completed Codex `agentMessage` items); `messages` has the raw rows of the turn, newest kept when a turn exceeds 1000 rows (`truncated` is set). `timeout_seconds` is a wall-clock bound: sleeps and requests are capped by it, nothing is requested once it has elapsed, and when it elapses `timed_out` is set and whatever arrived so far is returned, together with `next_after`, the ID of the last transcript row read. `timeout_seconds` defaults to 30 minutes and the block's own execution cap is two hours, so it is limited to 7200; a coding agent turn commonly runs 5-30 minutes, so one long wait is cheaper and more reliable than repeated short polls or scheduled follow-ups. `poll_interval_seconds` left at 0 scales with the timeout (one status check per 90 seconds of wait, between 10 and 60 seconds, and never more than half the wait so a short wait still checks the session). A wait that times out is continued, not restarted: pass `next_after` as `after` and `message_id` as `prompt_message_id` to Get Session with `wait_until_idle` on, which blocks until the agent finishes that prompt's turn and returns the newest messages after that row.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| session_id | Session to send the prompt to | str | Yes |
| message | Prompt for the agent | str | Yes |
| wait_for_reply | Wait until the agent is idle and return its reply. Coding agents typically run 5-30 minutes: prefer a single wait with a large timeout_seconds (max 7200) over repeated short polls or scheduled follow-ups. When timed_out is true the agent is still working; keep waiting with Conductor Get Session (wait_until_idle=true, after=next_after, prompt_message_id=the prompt's message id) rather than scheduling a check. | bool | No |
| timeout_seconds | How long to wait, in seconds (max 7200). Coding agents typically run 5-30 minutes: prefer a single wait with a large timeout_seconds (max 7200) over repeated short polls or scheduled follow-ups. When timed_out is true the agent is still working; keep waiting with Conductor Get Session (wait_until_idle=true, after=next_after, prompt_message_id=the prompt's message id) rather than scheduling a check. | int | No |
| poll_interval_seconds | Seconds between status checks while waiting; 0 scales it with timeout_seconds (10-60s) | int | No |

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
| truncated | True when the turn produced more messages than are kept; messages holds the newest ones and reply may be incomplete | bool |
| next_after | ID of the last transcript row read while waiting; when timed_out is true pass it as after to Conductor Get Session with wait_until_idle (and the prompt's message id as prompt_message_id) to continue waiting from where this block stopped | str |
| error_message | Session error, if any | str |

### Possible use case
<!-- MANUAL: use_case -->
**Conversational driver**: Let AutoPilot steer a coding agent turn by turn: send a prompt, read `reply`, decide the next instruction.

**Long turn**: Set `timeout_seconds` to cover the whole task (up to 7200). If `timed_out` comes back true, call Get Session with `wait_until_idle`, `after` = `next_after` and `prompt_message_id` = `message_id` to keep waiting in one call instead of scheduling a follow-up.

**Fire and forget**: Turn `wait_for_reply` off to queue several prompts and check back later with Get Session.
<!-- END MANUAL -->

---
