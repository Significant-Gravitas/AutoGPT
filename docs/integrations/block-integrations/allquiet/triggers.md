# All Quiet Triggers
<!-- MANUAL: file_description -->
A webhook trigger that starts a graph when All Quiet posts an incident to it, so agents can react to incidents rather than only create them.
<!-- END MANUAL -->

## All Quiet Incident Trigger

### What it is
Triggers a graph when All Quiet posts an incident to this webhook

### How it works
<!-- MANUAL: how_it_works -->
Runs when All Quiet's outbound webhook posts to this block's URL. Because the request body is whatever Handlebars template you configure in All Quiet, the block does not impose a schema: it always emits the raw `payload`, and additionally reads the well-known keys from both All Quiet's stock flattened template (`incidentId`/`incidentTitle`/`incidentProperties`) and a fuller template that forwards the incident itself. If you enable signing in All Quiet and paste the secret into the block, each delivery is verified as an HMAC-SHA256 of `<timestamp>:<body>` — both the All Quiet (`x-aq-signature`) and AWS (`x-amzn-event-signature`) header formats are accepted — and stale timestamps are rejected to prevent replay. Leave the secret empty and the webhook URL is the only credential.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| signing_secret | Optional. If All Quiet's outbound webhook has signing enabled, paste the signing secret here and deliveries with a missing or bad signature are rejected. Both the All Quiet (x-aq-signature) and AWS (x-amzn-event-signature) formats are accepted. Leave empty for unsigned webhooks. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| payload | The raw payload All Quiet delivered | Dict[str, Any] |
| incident_id | ID of the incident that fired | str |
| incident_title | Title of the incident | str |
| event_id | ID of the timeline event that triggered this delivery | str |
| status | Incident status, if the payload template includes it | "Open" \| "Resolved" |
| severity | Incident severity, if the payload template includes it | "Critical" \| "Warning" \| "Minor" |
| attributes | Incident attributes flattened to a name/value mapping | Dict[str, str] |

### Possible use case
<!-- MANUAL: use_case -->
An incident is raised in All Quiet by a Prometheus alert. The trigger starts a graph that pulls the incident's attributes, queries the relevant logs and dashboards, drafts a probable-cause summary with an LLM, and posts it back as a comment — so the responder opens an incident that already has a first pass of investigation attached.
<!-- END MANUAL -->

---
