# Twitter Get List

### What it is
This block retrieves information about a specified Twitter List.

### What it does
This block retrieves information about a specified Twitter List.

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
| list_id | The ID of the List to lookup | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | ID of the Twitter List | str |
| name | Name of the Twitter List | str |
| owner_id | ID of the List owner | str |
| owner_username | Username of the List owner | str |
| data | Complete list data | Dict[str, True] |
| included | Additional data requested via expansions | Dict[str, True] |
| meta | Metadata about the response | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Get Owned Lists

### What it is
This block retrieves all Lists owned by a specified Twitter user.

### What it does
This block retrieves all Lists owned by a specified Twitter user.

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
| data | Complete owned lists data | List[Dict[str, True]] |
| included | Additional data requested via expansions | Dict[str, True] |
| meta | Metadata about the response | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
