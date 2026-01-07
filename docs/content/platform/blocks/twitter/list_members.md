# Twitter Add List Member

### What it is
This block adds a specified user to a Twitter List owned by the authenticated user.

### What it does
This block adds a specified user to a Twitter List owned by the authenticated user.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Get List Members

### What it is
This block retrieves the members of a specified Twitter List.

### What it does
This block retrieves the members of a specified Twitter List.

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
| data | Complete user data for list members | List[Dict[str, True]] |
| included | Additional data requested via expansions | Dict[str, True] |
| meta | Metadata including pagination info | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Get List Memberships

### What it is
This block retrieves all Lists that a specified user is a member of.

### What it does
This block retrieves all Lists that a specified user is a member of.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with your Twitter Lists:
- Select 'List_Owner_ID' to get details about who owns the list

This will let you see more details about the list owner when you also select user fields below. | ListExpansionsFilter | No |
| user_fields | Choose what information you want to see about list owners. This only works when you select 'List_Owner_ID' in expansions above.

You can see things like:
- Their username
- Profile picture
- Account details
- And more | TweetUserFieldsFilter | No |
| list_fields | Choose what information you want to see about the Twitter Lists themselves, such as:
- List name
- Description
- Number of followers
- Number of members
- Whether it's private
- Creation date
- And more | ListFieldsFilter | No |
| user_id | The ID of the user whose List memberships to retrieve | str | Yes |
| max_results | Maximum number of results per page (1-100) | int | No |
| pagination_token | Token for pagination of results | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| list_ids | List of list IDs | List[str] |
| next_token | Next token for pagination | str |
| data | List membership data | List[Dict[str, True]] |
| included | Additional data requested via expansions | Dict[str, True] |
| meta | Metadata about pagination | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Remove List Member

### What it is
This block removes a specified user from a Twitter List owned by the authenticated user.

### What it does
This block removes a specified user from a Twitter List owned by the authenticated user.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
