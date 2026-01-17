# Twitter Blocks
<!-- MANUAL: file_description -->
Blocks for managing blocked users on Twitter/X.
<!-- END MANUAL -->

## Twitter Get Blocked Users

### What it is
This block retrieves a list of users blocked by the authenticating user.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to retrieve a paginated list of users that the authenticated account has blocked. It authenticates using OAuth 2.0 with the required scopes (users.read, block.read) and sends a request to Twitter's blocked users endpoint.

The response includes user IDs and usernames by default, with optional expansions to include additional data like pinned tweets. Pagination is supported through tokens, allowing retrieval of large block lists in batches of up to 1,000 users per request.
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
| included | Additional data requested via expansions | Dict[str, Any] |
| meta | Metadata including pagination info | Dict[str, Any] |
| next_token | Next token for pagination | str |

### Possible use case
<!-- MANUAL: use_case -->
**Block List Audit**: Review your block list periodically to identify accounts you may want to unblock or to analyze blocking patterns.

**Safety Monitoring**: Track blocked accounts as part of a harassment monitoring workflow, documenting problematic accounts.

**Account Migration**: Export your block list when setting up a new account or for backup purposes.
<!-- END MANUAL -->

---
