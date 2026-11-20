# Conductor Routines
<!-- MANUAL: file_description -->
Creates a Conductor routine (a saved prompt that runs a fresh agent whenever its webhook is called) or rotates a routine's webhook secret. Requires your own Conductor API key, created at [app.conductor.build/users/api-keys](https://app.conductor.build/users/api-keys) and added through AutoGPT’s credentials UI. Select that credential for each Conductor block; no server-wide default key is used.
<!-- END MANUAL -->

## Conductor Manage Routine

### What it is
Create a Conductor routine (a saved prompt that runs a fresh agent in a project whenever its webhook URL is called) or rotate a routine's webhook secret. Returns the webhook URL.

### How it works
<!-- MANUAL: how_it_works -->
`create` posts `name`, `prompt`, `projectId`, `agent`, optional `model`/`effort`, `enabled` and a single webhook trigger to `POST /v0/routines`; `rotate_webhook_url` posts to `POST /v0/routines/{id}/rotate-secret`. Both responses include the trigger's `webhookUrl`, which the block surfaces as `webhook_url`; the plain routine listing in Get Account never includes it, so store the URL when you receive it. Rotating invalidates the previous URL.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| action | create a webhook-triggered routine, or rotate_webhook_url to replace an existing routine's webhook URL | "create" \| "rotate_webhook_url" | No |
| routine_id | Routine ID (rotate_webhook_url) | str | No |
| name | Routine name (create) | str | No |
| prompt | Prompt the agent runs each time the webhook fires (create) | str | No |
| project_id | Project (repository) the routine runs in (create). Find IDs with Get Account. | str | No |
| agent | Agent that runs the routine | "claude" \| "codex" \| "cursor" \| "acp" | No |
| model | Model id such as fable-5-1, opus-5-5-1m, sonnet-5-1m, gpt-6-astra or auto. Leave empty for Conductor's default. | str | No |
| effort | Reasoning effort; leave empty for the default | "" \| "none" \| "low" \| "medium" \| "high" \| "xhigh" \| "max" \| "ultra" | No |
| enabled | Whether the routine is enabled on creation | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| routine_id | ID of the routine | str |
| webhook_url | Webhook URL that triggers the routine. POST to it to run the routine; this is the only time the URL is shown. | str |
| routine | Full routine object | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Wire up automation**: Create a routine such as "triage new issues" and send its `webhook_url` to another system that calls it on events.

**Credential hygiene**: Rotate a leaked webhook secret and update the caller with the new URL.
<!-- END MANUAL -->

---
