# Twitter List Members
<!-- MANUAL: file_description -->
Blocks for managing members of Twitter/X lists.
<!-- END MANUAL -->

## Twitter Add List Member

### What it is
This block adds a specified user to a Twitter List owned by the authenticated user.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to add a user to a Twitter List that you own. The added user will appear in the list's member roster and their tweets will show in the list timeline.

The block authenticates using OAuth 2.0 with list write permissions. Only the list owner can add members. The target user does not need to approve being added to public lists.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list_id | The ID of the List to add the member to | str | Yes |
| user_id | The ID of the user to add to the List | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the member was successfully added | bool |

### Possible use case
<!-- MANUAL: use_case -->
**List Curation**: Build curated lists by adding relevant accounts you discover.

**Community Organization**: Add new community members or contributors to tracking lists.

**Research Lists**: Add accounts to research lists for ongoing monitoring projects.
<!-- END MANUAL -->

---

## Twitter Get List Members

### What it is
This block retrieves the members of a specified Twitter List.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve all members of a specific Twitter List. Results include user IDs, usernames, and optionally expanded profile data with pagination support.

The block uses Tweepy with OAuth 2.0 authentication. Works for public lists and private lists you own or are a member of. Returns member data in batches of up to 100 users.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with user data. Currently only 'pinned_tweet_id' is available to see a user's pinned tweet. | UserExpansionsFilter | No |
| tweet_fields | Select what tweet information you want to see in pinned tweets. This only works if you select 'pinned_tweet_id' in expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see, like username, bio, profile picture, etc. | TweetUserFieldsFilter | No |
| list_id | The ID of the List to get members from | str | Yes |
| max_results | Maximum number of results per page (1-100) | int | No |
| pagination_token | Token for pagination of results | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of member user IDs | List[str] |
| usernames | List of member usernames | List[str] |
| next_token | Next token for pagination | str |
| data | Complete user data for list members | List[Dict[str, Any]] |
| included | Additional data requested via expansions | Dict[str, Any] |
| meta | Metadata including pagination info | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Audience Analysis**: Analyze the composition of curated lists to understand target audiences.

**List Export**: Export list members for analysis or to recreate lists on other platforms.

**Member Verification**: Check if specific users are members of a particular list.
<!-- END MANUAL -->

---

## Twitter Get List Memberships

### What it is
This block retrieves all Lists that a specified user is a member of.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve all Lists that include a specific user as a member. This shows which curated lists have added this account.

The block uses Tweepy with OAuth 2.0 authentication and supports expansions for list owner data. Returns paginated results with list IDs and metadata. Only returns public lists unless querying your own memberships.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with your Twitter Lists: - Select 'List_Owner_ID' to get details about who owns the list  This will let you see more details about the list owner when you also select user fields below. | ListExpansionsFilter | No |
| user_fields | Choose what information you want to see about list owners. This only works when you select 'List_Owner_ID' in expansions above.  You can see things like: - Their username - Profile picture - Account details - And more | TweetUserFieldsFilter | No |
| list_fields | Choose what information you want to see about the Twitter Lists themselves, such as: - List name - Description - Number of followers - Number of members - Whether it's private - Creation date - And more | ListFieldsFilter | No |
| user_id | The ID of the user whose List memberships to retrieve | str | Yes |
| max_results | Maximum number of results per page (1-100) | int | No |
| pagination_token | Token for pagination of results | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| list_ids | List of list IDs | List[str] |
| next_token | Next token for pagination | str |
| data | List membership data | List[Dict[str, Any]] |
| included | Additional data requested via expansions | Dict[str, Any] |
| meta | Metadata about pagination | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Influence Analysis**: Discover what curated lists include an influencer or competitor.

**Profile Research**: Understand how others categorize a specific account.

**Visibility Assessment**: See which lists feature your own account for reputation tracking.
<!-- END MANUAL -->

---

## Twitter Remove List Member

### What it is
This block removes a specified user from a Twitter List owned by the authenticated user.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to remove a user from a Twitter List that you own. The removed user will no longer appear in the list and their tweets won't show in the list timeline.

The block authenticates using OAuth 2.0 with list write permissions. Only the list owner can remove members. The removed user is not notified.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list_id | The ID of the List to remove the member from | str | Yes |
| user_id | The ID of the user to remove from the List | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the member was successfully removed | bool |

### Possible use case
<!-- MANUAL: use_case -->
**List Maintenance**: Remove inactive or irrelevant accounts from your curated lists.

**Quality Control**: Remove accounts that no longer meet the list's criteria or purpose.

**List Cleanup**: Periodically clean up lists by removing accounts that have changed focus.
<!-- END MANUAL -->

---
