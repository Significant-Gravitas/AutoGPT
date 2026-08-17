# All Quiet Teams
<!-- MANUAL: file_description -->
A block for listing All Quiet teams. Mainly used to resolve the team IDs that the incident and on-call blocks accept as inputs.
<!-- END MANUAL -->

## All Quiet List Teams

### What it is
Lists All Quiet teams and their IDs

### How it works
<!-- MANUAL: how_it_works -->
Lists the teams in your All Quiet organization with their IDs, time zones and labels, optionally filtered by name. Its `team_ids` output is designed to feed the `team_ids` input on Create Incident, List Incidents and Get On-Call, so a graph can target teams by name without hardcoding UUIDs.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| display_name | Filter to teams whose name matches this text. | str | No |
| limit | Maximum number of teams to return. | int | No |
| offset | Number of teams to skip, for paging. | int | No |
| region | The All Quiet deployment your API key belongs to. Use EU if you signed up on allquiet.eu. | "us" \| "eu" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| teams | All matching teams | List[Team] |
| team | Each matching team, emitted one at a time | Team |
| team_ids | IDs of the matching teams, for use as team_ids elsewhere | List[str] |
| has_more | Whether more teams are available beyond this page | bool |

### Possible use case
<!-- MANUAL: use_case -->
A graph that should page whichever team owns a failing service looks the team up by display name, then passes the resulting ID to Create Incident — so renaming or re-creating a team in All Quiet doesn't require editing the agent.
<!-- END MANUAL -->

---
