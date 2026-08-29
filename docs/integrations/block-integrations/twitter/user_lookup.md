# Twitter User Lookup
<!-- MANUAL: file_description -->
Blocks for looking up Twitter/X user profiles and information.
<!-- END MANUAL -->

## Twitter Get User

### What it is
This block retrieves information about a specified Twitter user.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve detailed information about a single user. You can look up users either by their unique Twitter ID or by their username (handle). The block uses Tweepy with OAuth 2.0 authentication.

Optional expansions allow you to include additional data such as the user's pinned tweet. The response includes profile information like display name, bio, follower count, and profile image URL based on the user_fields selected.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with user data. Currently only 'pinned_tweet_id' is available to see a user's pinned tweet. | UserExpansionsFilter | No |
| tweet_fields | Select what tweet information you want to see in pinned tweets. This only works if you select 'pinned_tweet_id' in expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see, like username, bio, profile picture, etc. | TweetUserFieldsFilter | No |
| identifier | Choose whether to identify the user by their unique Twitter ID or by their username | Identifier | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | User ID | str |
| username_ | User username | str |
| name_ | User name | str |
| data | Complete user data | Dict[str, Any] |
| included | Additional data requested via expansions | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Profile Verification**: Look up a user's profile to verify their identity or check their account details before engaging.

**User Research**: Gather information about a specific account for competitive analysis or influencer research.

**Account Validation**: Verify that a username or user ID exists and is active before performing other operations.
<!-- END MANUAL -->

---

## Twitter Get Users

### What it is
This block retrieves information about multiple Twitter users.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve detailed information about multiple users in a single request. You can look up users by their Twitter IDs or usernames (handles), making it efficient for batch operations.

The block uses Tweepy with OAuth 2.0 authentication and supports expansions for additional data like pinned tweets. Returns arrays of user IDs, usernames, display names, and complete user data objects.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with user data. Currently only 'pinned_tweet_id' is available to see a user's pinned tweet. | UserExpansionsFilter | No |
| tweet_fields | Select what tweet information you want to see in pinned tweets. This only works if you select 'pinned_tweet_id' in expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see, like username, bio, profile picture, etc. | TweetUserFieldsFilter | No |
| identifier | Choose whether to identify users by their unique Twitter IDs or by their usernames | Identifier | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | User IDs | List[str] |
| usernames_ | User usernames | List[str] |
| names_ | User names | List[str] |
| data | Complete users data | List[Dict[str, Any]] |
| included | Additional data requested via expansions | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Batch Profile Lookup**: Retrieve profile information for a list of users at once, such as all participants in a conversation.

**Influencer Analysis**: Gather profile data for multiple influencers in a specific niche for comparison.

**Follow List Enrichment**: Get detailed information about accounts in your following or followers list.
<!-- END MANUAL -->

---
