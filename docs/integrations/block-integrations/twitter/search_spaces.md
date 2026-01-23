# Twitter Search Spaces
<!-- MANUAL: file_description -->
Blocks for searching Twitter/X Spaces live audio conversations.
<!-- END MANUAL -->

## Twitter Search Spaces

### What it is
This block searches for Twitter Spaces based on specified terms.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to search for Twitter Spaces (live audio conversations) matching a search term. Results can be filtered by state (live, scheduled, or all) and include Space metadata like title, host information, and participant counts.

The block uses Tweepy with OAuth 2.0 authentication and supports expansions to include additional data about creators, hosts, speakers, invited users, and topics. Returns paginated results with Space IDs, titles, and host information.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose additional information you want to get with your Twitter Spaces: - Select 'Invited_Users' to see who was invited - Select 'Speakers' to see who can speak - Select 'Creator' to get details about who made the Space - Select 'Hosts' to see who's hosting - Select 'Topics' to see Space topics | SpaceExpansionsFilter | No |
| space_fields | Choose what Space details you want to see, such as: - Title - Start/End times - Number of participants - Language - State (live/scheduled) - And more | SpaceFieldsFilter | No |
| user_fields | Choose what user information you want to see. This works when you select any of these in expansions above: - 'Creator' for Space creator details - 'Hosts' for host information - 'Speakers' for speaker details - 'Invited_Users' for invited user information | TweetUserFieldsFilter | No |
| query | Search term to find in Space titles | str | Yes |
| max_results | Maximum number of results to return (1-100) | int | No |
| state | Type of Spaces to return (live, scheduled, or all) | "live" \| "scheduled" \| "all" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of space IDs | List[str] |
| titles | List of space titles | List[str] |
| host_ids | List of host IDs | List[Any] |
| next_token | Next token for pagination | str |
| data | Complete space data | List[Dict[str, Any]] |
| includes | Additional data requested via expansions | Dict[str, Any] |
| meta | Metadata including pagination info | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Event Discovery**: Find live or upcoming Spaces about topics you're interested in to join or monitor.

**Industry Monitoring**: Track Spaces related to your industry to stay informed about discussions and trends.

**Competitor Analysis**: Search for Spaces hosted by competitors to understand their community engagement strategies.
<!-- END MANUAL -->

---
