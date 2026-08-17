# All Quiet Incident Search
<!-- MANUAL: file_description -->
Blocks that read All Quiet incidents — fetching a single incident by ID, or searching the incident list by status, severity, team, text or time range.
<!-- END MANUAL -->

## All Quiet Get Incident

### What it is
Fetches a single All Quiet incident by ID

### How it works
<!-- MANUAL: how_it_works -->
Fetches one incident by ID and flattens its current status and severity onto the result — All Quiet stores those at the head of the incident's event timeline rather than on the incident itself. `allowed_intents` reports which transitions the incident will currently accept. Turning on `include_markdown` makes a second call for All Quiet's rendered markdown report, which bundles the attributes and full timeline into prose that is well suited to feeding an LLM.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| incident_id | ID of the incident to fetch | str | Yes |
| include_markdown | Also fetch an LLM-friendly markdown report of the incident, including its attributes and full timeline. Costs one extra API call. | bool | No |
| region | The All Quiet deployment your API key belongs to. Use EU if you signed up on allquiet.eu. | "us" \| "eu" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| incident | The incident | Incident |
| status | Current status, when the incident reports one | "Open" \| "Resolved" |
| severity | Current severity, when the incident reports one | "Critical" \| "Warning" \| "Minor" |
| allowed_intents | Transitions this incident currently accepts | List[str] |
| markdown | Markdown report, if include_markdown was set | str |

### Possible use case
<!-- MANUAL: use_case -->
A triage agent receives an incident ID from the trigger block, fetches it with `include_markdown` enabled, and feeds the markdown report — attributes, timeline and all — to an LLM to draft a first-response summary and suggest a likely cause.
<!-- END MANUAL -->

---

## All Quiet List Incidents

### What it is
Searches All Quiet incidents by status, severity, team or text

### How it works
<!-- MANUAL: how_it_works -->
Searches the incident list with All Quiet's server-side filters: status, severity, teams, a free-text title match, an unattended flag, and a created-at range. Results are paginated (`limit`/`offset`) and `has_more` reports whether further pages exist. Alongside the full list the block emits each incident individually, plus a list of bare `incident_ids` that feeds straight into Get Incident or Update Incident.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| statuses | Only return incidents in these statuses. | List["Open" \| "Resolved"] | No |
| severities | Only return incidents at these severities. | List["Critical" \| "Warning" \| "Minor"] | No |
| team_ids | Only return incidents routed to these teams. | List[str] | No |
| search_term | Free-text match against the incident title. | str | No |
| unattended | Set true to return only incidents nobody has picked up. | bool | No |
| limit | Maximum number of incidents to return. | int | No |
| offset | Number of incidents to skip, for paging. | int | No |
| created_from | Only incidents created at or after this ISO-8601 timestamp. | str | No |
| created_until | Only incidents created at or before this ISO-8601 timestamp. | str | No |
| sort_by | Field to sort the results by. | "Created" \| "LastUpdatedAt" \| "LatestInteraction" \| "Urgency" \| "Title" | No |
| ascending | Sort oldest first instead of newest first. | bool | No |
| region | The All Quiet deployment your API key belongs to. Use EU if you signed up on allquiet.eu. | "us" \| "eu" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| incidents | All matching incidents | List[Incident] |
| incident | Each matching incident, emitted one at a time | Incident |
| incident_ids | IDs of the matching incidents | List[str] |
| count | Number of incidents returned | int |
| has_more | Whether more incidents are available beyond this page | bool |

### Possible use case
<!-- MANUAL: use_case -->
A morning-standup agent lists every incident created in the last 24 hours, groups them by severity, and posts a digest — or, filtering on `unattended`, chases anything still open that nobody has picked up.
<!-- END MANUAL -->

---
