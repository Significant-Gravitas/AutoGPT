# All Quiet Triggers
<!-- MANUAL: file_description -->
A webhook trigger that starts a graph when All Quiet posts an incident to it, so agents can react to incidents rather than only create them.
<!-- END MANUAL -->

## AllQuiet Incident Trigger

### What it is
Triggers a graph when All Quiet posts an incident to this webhook

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
