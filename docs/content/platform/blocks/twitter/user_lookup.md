# Twitter Get User

### What it is
This block retrieves information about a specified Twitter user.

### What it does
This block retrieves information about a specified Twitter user.

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
| identifier | Choose whether to identify the user by their unique Twitter ID or by their username | Identifier | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | User ID | str |
| username_ | User username | str |
| name_ | User name | str |
| data | Complete user data | Dict[str, True] |
| included | Additional data requested via expansions | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Get Users

### What it is
This block retrieves information about multiple Twitter users.

### What it does
This block retrieves information about multiple Twitter users.

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
| identifier | Choose whether to identify users by their unique Twitter IDs or by their usernames | Identifier | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | User IDs | List[str] |
| usernames_ | User usernames | List[str] |
| names_ | User names | List[str] |
| data | Complete users data | List[Dict[str, True]] |
| included | Additional data requested via expansions | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
