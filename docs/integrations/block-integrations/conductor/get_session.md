# Conductor Get Session
<!-- MANUAL: file_description -->
Reads a Conductor agent session: details, whether the agent is idle, working or errored, and recent transcript messages. Needs a `CONDUCTOR_API_KEY` credential, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys). Every Conductor block uses the same key.
<!-- END MANUAL -->

## Conductor Get Session

### What it is
Get a Conductor agent session: its details, whether the agent is idle, working or errored, and recent transcript messages.

### How it works
<!-- MANUAL: how_it_works -->
The block calls `GET /v0/sessions/{id}` and `GET /v0/sessions/{id}/status`, then `GET /v0/sessions/{id}/messages` with `limit=message_limit` and the optional `after` cursor (set `message_limit` to 0 to skip the transcript). `latest_reply` is the text of the most recent non-user message in that slice; message `content` is untyped upstream, so strings, `{text}` objects and lists of parts are all flattened. Pass `message_id` to also fetch one message via `GET /v0/messages/{id}`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| session_id | Session ID | str | Yes |
| message_limit | How many transcript messages to return; 0 skips the transcript | int | No |
| after | Only messages after this message ID | str | No |
| message_id | Also fetch this single message by ID | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| session | Session: id, name, model, resolvedModel, effort, fastMode, deepLink, archivedAt | Dict[str, Any] |
| status | idle, working or error | str |
| error_message | Last session error, if any | str |
| messages | Transcript messages: id, sessionIndex, type, content, receivedAt | List[Dict[str, Any]] |
| latest_reply | Text of the most recent agent message in the returned transcript slice | str |
| message | The single message requested by message_id | Dict[str, Any] |
| deep_link | Link that opens the session | str |

### Possible use case
<!-- MANUAL: use_case -->
**Poll a long task**: Check `status` on a schedule and read `latest_reply` once it turns `idle`.

**Incremental transcript**: Keep the last message id and pass it as `after` to fetch only what is new.
<!-- END MANUAL -->

---
