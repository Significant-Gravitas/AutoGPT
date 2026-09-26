# Conductor Search
<!-- MANUAL: file_description -->
Runs a read-only SQL query over the transcripts of the Conductor workspaces you can access. Needs a `CONDUCTOR_API_KEY` credential, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys). Every Conductor block uses the same key.
<!-- END MANUAL -->

## Conductor Search Transcripts

### What it is
Search Conductor session transcripts with a read-only SQL query. Useful for finding what an agent said or did across workspaces.

### How it works
<!-- MANUAL: how_it_works -->
The block posts `{query}` to `POST /v0/sql`. Conductor executes it read-only and returns `rows` (objects keyed by column), `row_count` and `truncated` when the server cut the result short (at most 500 rows come back). The only queryable relation is `session_transcripts_view`, one row per session with `session_id`, `workspace_id`, `transcript` (plain text of the conversation), `session_title`, `agent_type`, `model`, `workspace_name`, `workspace_state`, `repo_url`, `session_created_at`, `transcript_updated_at`, `workspace_created_at`, `workspace_creator_id` and `workspace_creator_name`; any other table, view or function is rejected with a 400 that comes back as a block error carrying Conductor's message. Keep queries bounded with `LIMIT`, for example `SELECT session_id, session_title, workspace_name FROM session_transcripts_view WHERE transcript ILIKE '%database migration%' ORDER BY transcript_updated_at DESC LIMIT 20`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | A single read-only SELECT over session_transcripts_view, the only queryable view (one row per session of the workspaces you can access). Columns: session_id, workspace_id, transcript (plain text of the conversation), session_title, agent_type, model, workspace_name, workspace_state, repo_url, session_created_at, transcript_updated_at, workspace_created_at, workspace_creator_id, workspace_creator_name. At most 500 rows are returned, so add a LIMIT. Example: SELECT session_id, session_title, workspace_name FROM session_transcripts_view WHERE transcript ILIKE '%database migration%' ORDER BY transcript_updated_at DESC LIMIT 20 | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| rows | Result rows as objects | List[Dict[str, Any]] |
| row_count | Number of rows returned | int |
| truncated | True when the server cut the result short | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Find prior art**: Search past sessions for how an agent solved a similar bug before starting a new workspace.

**Reporting**: Count sessions per repository or per model over the last week for a status summary.
<!-- END MANUAL -->

---
