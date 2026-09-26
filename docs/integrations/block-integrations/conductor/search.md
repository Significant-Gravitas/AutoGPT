# Conductor Search
<!-- MANUAL: file_description -->
Runs a read-only SQL query over the transcripts of the Conductor workspaces you can access. Needs a `CONDUCTOR_API_KEY` credential, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys). Every Conductor block uses the same key.
<!-- END MANUAL -->

## Conductor Search Transcripts

### What it is
Search Conductor session transcripts with a read-only SQL query. Useful for finding what an agent said or did across workspaces.

### How it works
<!-- MANUAL: how_it_works -->
The block posts `{query}` to `POST /v0/sql`. Conductor executes it read-only against the transcript store and returns `rows` (objects keyed by column), `row_count` and `truncated` when the server cut the result short. Keep queries bounded with `LIMIT`; syntax errors and rejected statements come back as a block error carrying Conductor's message.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Read-only SQL over the transcripts of the workspaces you can access, e.g. SELECT * FROM messages WHERE content LIKE '%bug%' LIMIT 20 | str | Yes |

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
