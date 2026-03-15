# Twitter Mutes
<!-- MANUAL: file_description -->
Blocks for managing muted users on Twitter/X.
<!-- END MANUAL -->

## Twitter Get Muted Users

### What it is
This block gets a list of users muted by the authenticating user.

### How it works
<!-- MANUAL: how_it_works -->
This block connects to the Twitter API v2 via Tweepy to retrieve all users that the authenticated account has muted. It uses OAuth 2.0 authentication with appropriate scopes and returns paginated results with user IDs, usernames, and optional expanded data.

The mute list is returned in batches (default 100, up to 1,000 per page), with pagination tokens for navigating large lists. Unlike blocking, muted users can still see and interact with your content—they're simply hidden from your timeline.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with user data. Currently only 'pinned_tweet_id' is available to see a user's pinned tweet. | UserExpansionsFilter | No |
| tweet_fields | Select what tweet information you want to see in pinned tweets. This only works if you select 'pinned_tweet_id' in expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see, like username, bio, profile picture, etc. | TweetUserFieldsFilter | No |
| max_results | The maximum number of results to be returned per page (1-1000). Default is 100. | int | No |
| pagination_token | Token to request next/previous page of results | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of muted user IDs | List[str] |
| usernames | List of muted usernames | List[str] |
| next_token | Next token for pagination | str |
| data | Complete user data for muted users | List[Dict[str, Any]] |
| includes | Additional data requested via expansions | Dict[str, Any] |
| meta | Metadata including pagination info | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Timeline Cleanup**: Review which accounts you've muted to decide if you want to unmute any or convert mutes to blocks.

**Noise Reduction Audit**: Analyze your mute list to understand what types of content you're filtering from your feed.

**Account Management**: Export your mute list for backup or to apply similar muting patterns to another account.
<!-- END MANUAL -->

---

## Twitter Mute User

### What it is
This block mutes a specified Twitter user.

### How it works
<!-- MANUAL: how_it_works -->
This block sends a mute request to the Twitter API v2 for a specified user ID. The authenticated user will no longer see tweets from the muted account in their timeline, but the muted user is not notified and can still view and interact with your content.

The mute action is performed using Tweepy's client interface with OAuth 2.0 authentication. The block returns a success indicator confirming whether the mute was applied.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| target_user_id | The user ID of the user that you would like to mute | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the mute action was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Filtering**: Mute accounts that post frequently about topics you want to temporarily avoid without unfollowing.

**Event Management**: Mute accounts live-tweeting events you can't attend to avoid spoilers.

**Conversation Management**: Mute accounts during heated discussions without blocking them permanently.
<!-- END MANUAL -->

---

## Twitter Unmute User

### What it is
This block unmutes a specified Twitter user.

### How it works
<!-- MANUAL: how_it_works -->
This block sends an unmute request to the Twitter API v2 for a specified user ID. Once unmuted, tweets from that account will appear in your timeline again as normal.

The unmute action uses Tweepy's client interface with OAuth 2.0 authentication. The block returns a success indicator confirming whether the mute was removed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| target_user_id | The user ID of the user that you would like to unmute | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the unmute action was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Post-Event Restore**: Unmute accounts after an event ends that you were avoiding spoilers for.

**Periodic Review**: Unmute accounts as part of a scheduled review of your mute list.

**Relationship Repair**: Restore visibility of accounts after cooling-off periods from disagreements.
<!-- END MANUAL -->

---
