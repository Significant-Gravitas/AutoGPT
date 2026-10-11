# Conductor Get Session
<!-- MANUAL: file_description -->
Reads a Conductor agent session: details, whether the agent is idle, working or errored, and recent transcript messages. Requires your own Conductor API key, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys) and added through AutoGPT’s credentials UI. Select that credential for each Conductor block; no server-wide default key is used.
<!-- END MANUAL -->

## Conductor Get Session

### What it is
Get a Conductor agent session: its details, whether the agent is idle, working or errored, and recent transcript messages.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `GET /v0/sessions/{id}` and `GET /v0/sessions/{id}/status`, then reads the transcript through `GET /v0/sessions/{id}/messages`. The API only pages from the start in ascending order (100 rows per request at most), so by default the block locates the end of the transcript with cheap one-row probes and returns the newest `message_limit` messages; with `after` it instead reads the next `message_limit` messages following that message, which is how to poll incrementally (`next_after` is the row ID to pass on the next call). `after` takes a transcript row ID, such as `next_after` from Send Message, Create Session, Create Workspace or a previous Get Session call; the prompt ID those blocks return as `message_id` / `initial_message_id` is a different namespace that the API rejects with 404, so when it is passed the block locates the prompt's row itself (newest 300 rows, then up to 1000 older ones) and pages from there, reporting that row ID through `next_after` when nothing follows it yet. `has_more` reports whether older (default) or newer (`after`) messages exist beyond the slice; set `message_limit` to 0 to skip the transcript. `latest_reply` is the newest message with visible agent text, so trailing tool, lifecycle or status events do not hide the answer. Live rows are `userMessage` prompts and `agent` events wrapping the harness's raw payload; only Claude `assistant` text parts and completed Codex `agentMessage` items count as visible text. Pass `message_id` to also fetch one message via `GET /v0/messages/{id}` (a transcript row ID, not a send receipt).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| session_id | Session ID | str | Yes |
| message_limit | How many transcript messages to return: the most recent ones, or the ones following `after` when it is set; 0 skips the transcript | int | No |
| after | Read forward from this message (exclusive) instead of returning the most recent messages. Accepts a transcript row ID (next_after from a previous call or from Send Message / Create Session) or the prompt ID those blocks return as message_id / initial_message_id, which is resolved to the prompt's row | str | No |
| message_id | Also fetch this single message by its transcript row ID (not a prompt ID returned by Send Message) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| session | Session: id, name, model, resolvedModel, effort, fastMode, deepLink, archivedAt | Dict[str, Any] |
| status | idle, working or error | str |
| error_message | Last session error, if any | str |
| messages | Transcript messages, oldest first: id, sessionIndex, type, content, receivedAt | List[Dict[str, Any]] |
| latest_reply | Text of the newest agent message with visible text in the returned transcript slice | str |
| has_more | True when the transcript has messages beyond the returned slice: older ones by default, newer ones when after is set | bool |
| next_after | Transcript row ID of the last returned message (or the row `after` resolved to when nothing followed it); pass it as after to read what follows | str |
| message | The single message requested by message_id | Dict[str, Any] |
| deep_link | Link that opens the session | str |

### Possible use case
<!-- MANUAL: use_case -->
**Poll a long task**: Check `status` on a schedule and read `latest_reply` once it turns `idle`.

**Incremental transcript**: Keep the last message id and pass it as `after` to fetch only what is new.
<!-- END MANUAL -->

---
