# Twitter List Lookup
<!-- MANUAL: file_description -->
Blocks for retrieving information about Twitter/X lists.
<!-- END MANUAL -->

## Twitter Get List

### What it is
This block retrieves information about a specified Twitter List.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve detailed information about a specific Twitter List by its ID. Returns list metadata including name, description, member count, follower count, and privacy status.

The block uses Tweepy with OAuth 2.0 authentication and supports expansions to include owner profile data. Works for both public lists and private lists you own or follow.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with your Twitter Lists: - Select 'List_Owner_ID' to get details about who owns the list  This will let you see more details about the list owner when you also select user fields below. | ListExpansionsFilter | No |
| user_fields | Choose what information you want to see about list owners. This only works when you select 'List_Owner_ID' in expansions above.  You can see things like: - Their username - Profile picture - Account details - And more | TweetUserFieldsFilter | No |
| list_fields | Choose what information you want to see about the Twitter Lists themselves, such as: - List name - Description - Number of followers - Number of members - Whether it's private - Creation date - And more | ListFieldsFilter | No |
| list_id | The ID of the List to lookup | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | ID of the Twitter List | str |
| name | Name of the Twitter List | str |
| owner_id | ID of the List owner | str |
| owner_username | Username of the List owner | str |
| data | Complete list data | Dict[str, Any] |
| included | Additional data requested via expansions | Dict[str, Any] |
| meta | Metadata about the response | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**List Verification**: Verify a list exists and check its current details before performing operations on it.

**List Discovery**: Retrieve information about interesting lists to decide whether to follow them.

**Analytics Preparation**: Gather list metadata for reporting or analysis purposes.
<!-- END MANUAL -->

---

## Twitter Get Owned Lists

### What it is
This block retrieves all Lists owned by a specified Twitter user.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve all Twitter Lists created by a specific user. Results include list IDs, names, and detailed metadata with pagination support for users with many lists.

The block uses Tweepy with OAuth 2.0 authentication and supports expansions to include owner profile data. Only returns public lists unless querying your own account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with your Twitter Lists: - Select 'List_Owner_ID' to get details about who owns the list  This will let you see more details about the list owner when you also select user fields below. | ListExpansionsFilter | No |
| user_fields | Choose what information you want to see about list owners. This only works when you select 'List_Owner_ID' in expansions above.  You can see things like: - Their username - Profile picture - Account details - And more | TweetUserFieldsFilter | No |
| list_fields | Choose what information you want to see about the Twitter Lists themselves, such as: - List name - Description - Number of followers - Number of members - Whether it's private - Creation date - And more | ListFieldsFilter | No |
| user_id | The user ID whose owned Lists to retrieve | str | Yes |
| max_results | Maximum number of results per page (1-100) | int | No |
| pagination_token | Token for pagination | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| list_ids | List ids of the owned lists | List[str] |
| list_names | List names of the owned lists | List[str] |
| next_token | Token for next page of results | str |
| data | Complete owned lists data | List[Dict[str, Any]] |
| included | Additional data requested via expansions | Dict[str, Any] |
| meta | Metadata about the response | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**List Discovery**: Find all public lists created by influencers or thought leaders in your industry.

**Account Audit**: Review all lists you own to identify ones to update or delete.

**Competitive Analysis**: Discover how competitors organize their Twitter following through their public lists.
<!-- END MANUAL -->

---
