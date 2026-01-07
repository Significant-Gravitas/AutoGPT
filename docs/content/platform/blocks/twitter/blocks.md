# Twitter Geted Users

### What it is
This block retrieves a list of users blocked by the authenticating user.

### What it does
This block retrieves a list of users blocked by the authenticating user.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with user data. Currently only 'pinned_tweet_id' is available to see a user's pinned tweet. | UserExpansionsFilter | No |
| tweet_fields | Select what tweet information you want to see in pinned tweets. This only works if you select 'pinned_tweet_id' in expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see, like username, bio, profile picture, etc. | TweetUserFieldsFilter | No |
| max_results | Maximum number of results to return (1-1000, default 100) | int | No |
| pagination_token | Token for retrieving next/previous page of results | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| user_ids | List of blocked user IDs | List[str] |
| usernames_ | List of blocked usernames | List[str] |
| included | Additional data requested via expansions | Dict[str, True] |
| meta | Metadata including pagination info | Dict[str, True] |
| next_token | Next token for pagination | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
