# All Quiet Incident Search
<!-- MANUAL: file_description -->
Blocks that read All Quiet incidents — fetching a single incident by ID, or searching the incident list by status, severity, team, text or time range.
<!-- END MANUAL -->

## AllQuiet Get Incident

### What it is
Fetches a single All Quiet incident by ID

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## AllQuiet List Incidents

### What it is
Searches All Quiet incidents by status, severity, team or text

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
