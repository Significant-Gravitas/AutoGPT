# Twitter Get Muted Users

### What it is
This block gets a list of users muted by the authenticating user.

### What it does
This block gets a list of users muted by the authenticating user.

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
| max_results | The maximum number of results to be returned per page (1-1000). Default is 100. | int | No |
| pagination_token | Token to request next/previous page of results | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of muted user IDs | List[str] |
| usernames | List of muted usernames | List[str] |
| next_token | Next token for pagination | str |
| data | Complete user data for muted users | List[Dict[str, True]] |
| includes | Additional data requested via expansions | Dict[str, True] |
| meta | Metadata including pagination info | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Mute User

### What it is
This block mutes a specified Twitter user.

### What it does
This block mutes a specified Twitter user.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Unmute User

### What it is
This block unmutes a specified Twitter user.

### What it does
This block unmutes a specified Twitter user.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
