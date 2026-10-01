# Conductor Get Session
<!-- MANUAL: file_description -->
Reads a Conductor agent session: details, whether the agent is idle, working or errored, and recent transcript messages. Requires your own Conductor API key, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys) and added through AutoGPT’s credentials UI. Select that credential for each Conductor block; no server-wide default key is used.
<!-- END MANUAL -->

## Conductor Get Session

### What it is
Get a Conductor agent session: its details, whether the agent is idle, working or errored, and recent transcript messages. With wait_until_idle it blocks in-tool until the agent finishes, which is how to keep waiting after a Send Message or Create Session wait timed out (after=next_after).

### How it works
<!-- MANUAL: how_it_works -->
The block calls `GET /v0/sessions/{id}` and `GET /v0/sessions/{id}/status`, then reads the transcript through `GET /v0/sessions/{id}/messages`. The API only pages from the start in ascending order (100 rows per request at most), so by default the block locates the end of the transcript with cheap one-row probes and returns the newest `message_limit` messages; with `after` it instead reads the next `message_limit` messages following that message ID, which is how to poll incrementally (`next_after` is the ID to pass on the next call). `has_more` reports whether older (default) or newer (`after`) messages exist beyond the slice; set `message_limit` to 0 to skip the transcript. `latest_reply` is the newest message with visible agent text, so trailing tool, lifecycle or status events do not hide the answer. Live rows are `userMessage` prompts and `agent` events wrapping the harness's raw payload; only Claude `assistant` text parts and completed Codex `agentMessage` items count as visible text. Pass `message_id` to also fetch one message via `GET /v0/messages/{id}` (a transcript row ID, not a send receipt). With `wait_until_idle` the block first polls `GET /v0/sessions/{id}/status` every `poll_interval_seconds` (0 scales it with the timeout) until the session is `idle` or `error`, for at most `timeout_seconds` (default 30 minutes, max 7200); a session that is already idle costs one request, and `timed_out` is set when the deadline passed first. It then reads the transcript as usual, except that with `after` it returns the newest `message_limit` messages following that row rather than the first ones, so `latest_reply` holds the answer of a long turn without paging (`has_more` then means older rows after `after` were skipped). This is how to continue a Send Message or Create Session wait that timed out: pass its `next_after` as `after`. If this wait times out too, call again with `after` = `next_after`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| session_id | Session ID | str | Yes |
| message_limit | How many transcript messages to return: the most recent ones, or the ones following `after` when it is set; 0 skips the transcript | int | No |
| after | Read forward from this transcript message ID (exclusive) instead of returning the most recent messages; use next_after from a previous call to poll incrementally | str | No |
| message_id | Also fetch this single message by ID | str | No |
| wait_until_idle | Block until the session is idle or errored before reading it, instead of polling from outside. Use this to keep waiting after a Send Message or Create Session wait timed out: pass its next_after as after and the newest messages following it are returned. Returns at once when the session is already idle. | bool | No |
| timeout_seconds | How long wait_until_idle waits, in seconds (max 7200). Coding agents typically run 5-30 minutes: prefer a single long wait over repeated short polls or scheduled follow-ups; if timed_out is true, call again with after=next_after to keep waiting. | int | No |
| poll_interval_seconds | Seconds between status checks while waiting; 0 scales it with timeout_seconds (10-60s) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| session | Session: id, name, model, resolvedModel, effort, fastMode, deepLink, archivedAt | Dict[str, Any] |
| status | idle, working or error | str |
| error_message | Last session error, if any | str |
| messages | Transcript messages, oldest first: id, sessionIndex, type, content, receivedAt | List[Dict[str, Any]] |
| latest_reply | Text of the newest agent message with visible text in the returned transcript slice | str |
| has_more | True when the transcript has messages beyond the returned slice: older ones by default, newer ones when after is set, and with wait_until_idle older ones that followed after | bool |
| next_after | ID of the last returned message; pass it as after to read what follows, or to continue a wait that timed out | str |
| timed_out | True when wait_until_idle was set and the session was still working when timeout_seconds elapsed; call again with after=next_after to keep waiting | bool |
| message | The single message requested by message_id | Dict[str, Any] |
| deep_link | Link that opens the session | str |

### Possible use case
<!-- MANUAL: use_case -->
**Wait for a long task**: Turn on `wait_until_idle` with a `timeout_seconds` that covers the task and read `latest_reply` when it returns; coding agents typically run 5-30 minutes, so one wait beats a schedule of short checks.

**Continue a timed-out wait**: Pass the `next_after` from a timed-out Send Message or Create Session as `after` with `wait_until_idle` on to pick the turn up where that block stopped.

**Incremental transcript**: Keep the last message id and pass it as `after` to fetch only what is new.
<!-- END MANUAL -->

---
