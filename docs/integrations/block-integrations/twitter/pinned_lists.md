# Twitter Pinned Lists
<!-- MANUAL: file_description -->
Blocks for managing pinned lists on Twitter/X.
<!-- END MANUAL -->

## Twitter Get Pinned Lists

### What it is
This block returns the Lists pinned by the authenticated user.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve all Lists that the authenticated user has pinned for quick access. Pinned lists appear prominently in the user's Lists tab on Twitter.

The block uses Tweepy with OAuth 2.0 authentication and supports expansions to include owner profile data and detailed list metadata. Returns list IDs, names, and complete list data objects.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with your Twitter Lists: - Select 'List_Owner_ID' to get details about who owns the list  This will let you see more details about the list owner when you also select user fields below. | ListExpansionsFilter | No |
| user_fields | Choose what information you want to see about list owners. This only works when you select 'List_Owner_ID' in expansions above.  You can see things like: - Their username - Profile picture - Account details - And more | TweetUserFieldsFilter | No |
| list_fields | Choose what information you want to see about the Twitter Lists themselves, such as: - List name - Description - Number of followers - Number of members - Whether it's private - Creation date - And more | ListFieldsFilter | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| list_ids | List IDs of the pinned lists | List[str] |
| list_names | List names of the pinned lists | List[str] |
| data | Response data containing pinned lists | List[Dict[str, Any]] |
| included | Additional data requested via expansions | Dict[str, Any] |
| meta | Metadata about the response | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Workflow Configuration**: Retrieve pinned lists to understand which lists are prioritized in user workflows.

**Settings Backup**: Export pinned list configurations for backup or account migration purposes.

**Dashboard Setup**: Identify pinned lists to build monitoring dashboards around priority content sources.
<!-- END MANUAL -->

---

## Twitter Pin List

### What it is
This block allows the authenticated user to pin a specified List.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to pin a Twitter List for quick access. Pinned lists appear at the top of your Lists tab on Twitter for easy navigation.

The block authenticates using OAuth 2.0 with list write permissions and sends a POST request to create the pin relationship. You can pin both your own lists and lists created by others.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list_id | The ID of the List to pin | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the pin was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Priority Organization**: Pin important lists to ensure they're easily accessible for daily monitoring.

**Workflow Setup**: Automatically pin newly created lists as part of project initialization workflows.

**Quick Access Configuration**: Pin frequently used lists to streamline content discovery.
<!-- END MANUAL -->

---

## Twitter Unpin List

### What it is
This block allows the authenticated user to unpin a specified List.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to unpin a Twitter List from your quick access area. The list remains followed but no longer appears in your pinned section.

The block authenticates using OAuth 2.0 with list write permissions and sends a DELETE request to remove the pin relationship. Returns a success indicator confirming the list was unpinned.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| list_id | The ID of the List to unpin | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the unpin was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Pin Management**: Unpin lists that are no longer priorities to make room for more relevant ones.

**Workflow Cleanup**: Remove pins from project-specific lists after campaigns or initiatives conclude.

**Organization Maintenance**: Periodically unpin outdated lists to keep your pinned section focused.
<!-- END MANUAL -->

---
