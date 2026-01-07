# Twitter Search Spaces

### What it is
This block searches for Twitter Spaces based on specified terms.

### What it does
This block searches for Twitter Spaces based on specified terms.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose additional information you want to get with your Twitter Spaces:
- Select 'Invited_Users' to see who was invited
- Select 'Speakers' to see who can speak
- Select 'Creator' to get details about who made the Space
- Select 'Hosts' to see who's hosting
- Select 'Topics' to see Space topics | SpaceExpansionsFilter | No |
| space_fields | Choose what Space details you want to see, such as:
- Title
- Start/End times
- Number of participants
- Language
- State (live/scheduled)
- And more | SpaceFieldsFilter | No |
| user_fields | Choose what user information you want to see. This works when you select any of these in expansions above:
- 'Creator' for Space creator details
- 'Hosts' for host information
- 'Speakers' for speaker details
- 'Invited_Users' for invited user information | TweetUserFieldsFilter | No |
| query | Search term to find in Space titles | str | Yes |
| max_results | Maximum number of results to return (1-100) | int | No |
| state | Type of Spaces to return (live, scheduled, or all) | "live" | "scheduled" | "all" | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of space IDs | List[str] |
| titles | List of space titles | List[str] |
| host_ids | List of host IDs | List[Any] |
| next_token | Next token for pagination | str |
| data | Complete space data | List[Dict[str, True]] |
| includes | Additional data requested via expansions | Dict[str, True] |
| meta | Metadata including pagination info | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
