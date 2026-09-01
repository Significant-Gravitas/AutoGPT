# All Quiet On Call
<!-- MANUAL: file_description -->
A block for reading All Quiet's on-call rotation: who is responsible right now, or who was (or will be) at any other point in time.
<!-- END MANUAL -->

## AllQuiet Get On Call

### What it is
Looks up who is on call in All Quiet, now or at a given time

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| team_ids | Limit to these teams. Leave empty for every team. | List[str] | No |
| user_ids | Limit to these users. Leave empty for every user. | List[str] | No |
| timestamp | ISO-8601 timestamp to evaluate the rotation at. Leave empty for right now. | str | No |
| region | The All Quiet deployment your API key belongs to. Use EU if you signed up on allquiet.eu. | "us" \| "eu" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| shifts | Every matching on-call assignment | List[OnCallShift] |
| shift | Each on-call assignment, emitted one at a time | OnCallShift |
| users | The on-call users, deduplicated across teams | List[AllQuietUser] |
| user_ids | IDs of the on-call users | List[str] |
| emails | Email addresses of the on-call users | List[str] |
| users_without_email | On-call users carrying no email address. These are counted in users/has_coverage but absent from emails, so a graph that notifies by email alone would silently skip them | List[AllQuietUser] |
| has_coverage | False when nobody is on call for the requested teams/time, so a graph can branch to a fallback instead of silently paging no one | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
