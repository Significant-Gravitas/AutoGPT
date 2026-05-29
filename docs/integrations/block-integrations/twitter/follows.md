# Twitter Follows
<!-- MANUAL: file_description -->
Blocks for following and unfollowing users on Twitter/X.
<!-- END MANUAL -->

## Twitter Follow User

### What it is
This block follows a specified Twitter user.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to create a follow relationship from the authenticated user to the specified target user. The follow action is public—the target user will be notified and can see that you followed them.

The block authenticates using OAuth 2.0 with follow write permissions. If the target user has a protected account, a follow request is sent instead of an immediate follow. Returns a success indicator confirming the action.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| target_user_id | The user ID of the user that you would like to follow | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the follow action was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Influencer Engagement**: Automatically follow industry influencers or thought leaders you want to engage with.

**Community Building**: Follow users who interact with your content to build reciprocal relationships.

**Network Expansion**: Follow users in specific niches or communities to expand your network strategically.
<!-- END MANUAL -->

---

## Twitter Get Followers

### What it is
This block retrieves followers of a specified Twitter user.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve a paginated list of users who follow a specified account. Results include user IDs, usernames, and optionally expanded profile data.

The block uses Tweepy with OAuth 2.0 authentication. Followers are returned in reverse chronological order (most recent first), with pagination support for accounts with many followers. Expansions can include pinned tweet data for each follower.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with user data. Currently only 'pinned_tweet_id' is available to see a user's pinned tweet. | UserExpansionsFilter | No |
| tweet_fields | Select what tweet information you want to see in pinned tweets. This only works if you select 'pinned_tweet_id' in expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see, like username, bio, profile picture, etc. | TweetUserFieldsFilter | No |
| target_user_id | The user ID whose followers you would like to retrieve | str | Yes |
| max_results | Maximum number of results to return (1-1000, default 100) | int | No |
| pagination_token | Token for retrieving next/previous page of results | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of follower user IDs | List[str] |
| usernames | List of follower usernames | List[str] |
| next_token | Next token for pagination | str |
| data | Complete user data for followers | List[Dict[str, Any]] |
| includes | Additional data requested via expansions | Dict[str, Any] |
| meta | Metadata including pagination info | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Audience Analysis**: Analyze the followers of a competitor or influencer to understand their audience demographics.

**Follower Monitoring**: Track new followers over time to identify growth patterns or notable new followers.

**Engagement Targeting**: Identify active followers for targeted engagement or outreach campaigns.
<!-- END MANUAL -->

---

## Twitter Get Following

### What it is
This block retrieves the users that a specified Twitter user is following.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve a paginated list of users that a specified account follows. Results include user IDs, usernames, and optionally expanded profile data.

The block uses Tweepy with OAuth 2.0 authentication. Following lists are returned with pagination support for accounts following many users. Expansions can include pinned tweet data for each followed account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with user data. Currently only 'pinned_tweet_id' is available to see a user's pinned tweet. | UserExpansionsFilter | No |
| tweet_fields | Select what tweet information you want to see in pinned tweets. This only works if you select 'pinned_tweet_id' in expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see, like username, bio, profile picture, etc. | TweetUserFieldsFilter | No |
| target_user_id | The user ID whose following you would like to retrieve | str | Yes |
| max_results | Maximum number of results to return (1-1000, default 100) | int | No |
| pagination_token | Token for retrieving next/previous page of results | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of following user IDs | List[str] |
| usernames | List of following usernames | List[str] |
| next_token | Next token for pagination | str |
| data | Complete user data for following | List[Dict[str, Any]] |
| includes | Additional data requested via expansions | Dict[str, Any] |
| meta | Metadata including pagination info | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Interest Analysis**: Analyze who an influencer or competitor follows to understand their interests and network.

**Discover Accounts**: Find relevant accounts to follow by examining the following lists of users in your niche.

**Relationship Mapping**: Map professional networks by analyzing mutual follows and connections.
<!-- END MANUAL -->

---

## Twitter Unfollow User

### What it is
This block unfollows a specified Twitter user.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to remove a follow relationship from the authenticated user to the specified target user. The unfollow is processed silently—the target user is not notified.

The block authenticates using OAuth 2.0 with follow write permissions and sends a DELETE request to remove the follow relationship. Returns a success indicator confirming the unfollow was processed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| target_user_id | The user ID of the user that you would like to unfollow | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the unfollow action was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Account Cleanup**: Unfollow inactive accounts or accounts that no longer post relevant content.

**Feed Curation**: Unfollow accounts to reduce noise in your timeline and focus on important content.

**Following List Management**: Maintain a manageable following count by periodically unfollowing accounts.
<!-- END MANUAL -->

---
