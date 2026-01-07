# Twitter Get Pinned Lists

### What it is
This block returns the Lists pinned by the authenticated user.

### What it does
This block returns the Lists pinned by the authenticated user.

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

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| list_ids | List IDs of the pinned lists | List[str] |
| list_names | List names of the pinned lists | List[str] |
| data | Response data containing pinned lists | List[Dict[str, True]] |
| included | Additional data requested via expansions | Dict[str, True] |
| meta | Metadata about the response | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Pin List

### What it is
This block allows the authenticated user to pin a specified List.

### What it does
This block allows the authenticated user to pin a specified List.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Unpin List

### What it is
This block allows the authenticated user to unpin a specified List.

### What it does
This block allows the authenticated user to unpin a specified List.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
