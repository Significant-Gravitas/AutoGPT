# All Quiet Teams
<!-- MANUAL: file_description -->
A block for listing All Quiet teams. Mainly used to resolve the team IDs that the incident and on-call blocks accept as inputs.
<!-- END MANUAL -->

## AllQuiet List Teams

### What it is
Lists All Quiet teams and their IDs

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
