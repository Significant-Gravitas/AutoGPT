# All Quiet Incidents
<!-- MANUAL: file_description -->
Blocks that create All Quiet incidents and move them through their lifecycle. Use these when an agent needs to raise an alert that reaches a human, or to acknowledge, resolve, escalate or comment on one it already raised.
<!-- END MANUAL -->

## AllQuiet Create Incident

### What it is
Creates an incident in All Quiet and pages the on-call responder

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| title | Short summary of what is wrong. Shown in every alert. | str | Yes |
| severity | How urgent the incident is. | "Critical" \| "Warning" \| "Minor" | No |
| status | Open pages the on-call responder. Resolved records the incident without paging anyone. | "Open" \| "Resolved" | No |
| message | Longer description with context for the responder. | str | No |
| team_ids | Teams to route the incident to. Leave empty to use the integration's default routing. | List[str] | No |
| service_ids | Affected services, used for status pages and uptime. | List[str] | No |
| user_ids | Users to assign directly, in addition to on-call routing. | List[str] | No |
| attributes | Extra key/value context, e.g. host or runbook URL. | Dict[str, str] | No |
| message_is_public | Show the message on a connected public status page. | bool | No |
| region | The All Quiet deployment your API key belongs to. Use EU if you signed up on allquiet.eu. | "us" \| "eu" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| incident | The incident that was created | Incident |
| incident_id | ID of the new incident, for later get/update calls | str |
| on_call_users | Users the incident was routed to. Often empty in the create response because All Quiet resolves routing asynchronously — read the incident back with Get Incident to see the responders. | List[AllQuietUser] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AllQuiet Update Incident

### What it is
Investigates, resolves, escalates or comments on an All Quiet incident

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| incident_id | ID of the incident to update | str | Yes |
| intent | The transition to apply. An incident only accepts the intents listed in its allowed_intents — e.g. Investigated/Resolved on an open incident, Unresolved on a resolved one. | "Investigated" \| "Resolved" \| "Unresolved" \| "Escalated" \| "Commented" \| "Snoozed" \| "Archived" | No |
| message | Note recorded on the incident timeline with this change. | str | No |
| severity | Optionally change the severity at the same time. | "Critical" \| "Warning" \| "Minor" | No |
| message_is_public | Show the message on a connected public status page. | bool | No |
| region | The All Quiet deployment your API key belongs to. Use EU if you signed up on allquiet.eu. | "us" \| "eu" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| incident | The incident after the update | Incident |
| allowed_intents | Transitions the incident accepts after this update, so a graph can pick its next intent without re-reading the incident | List[str] |
| status | Status after the update, when the incident reports one | "Open" \| "Resolved" |
| severity | Severity after the update, when the incident reports one | "Critical" \| "Warning" \| "Minor" |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
