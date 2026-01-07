# Twitter Follow User

### What it is
This block follows a specified Twitter user.

### What it does
This block follows a specified Twitter user.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| target_user_id | The user ID of the user that you would like to follow | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the follow action was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Get Followers

### What it is
This block retrieves followers of a specified Twitter user.

### What it does
This block retrieves followers of a specified Twitter user.

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
| target_user_id | The user ID whose followers you would like to retrieve | str | Yes |
| max_results | Maximum number of results to return (1-1000, default 100) | int | No |
| pagination_token | Token for retrieving next/previous page of results | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of follower user IDs | List[str] |
| usernames | List of follower usernames | List[str] |
| next_token | Next token for pagination | str |
| data | Complete user data for followers | List[Dict[str, True]] |
| includes | Additional data requested via expansions | Dict[str, True] |
| meta | Metadata including pagination info | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Get Following

### What it is
This block retrieves the users that a specified Twitter user is following.

### What it does
This block retrieves the users that a specified Twitter user is following.

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
| target_user_id | The user ID whose following you would like to retrieve | str | Yes |
| max_results | Maximum number of results to return (1-1000, default 100) | int | No |
| pagination_token | Token for retrieving next/previous page of results | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of following user IDs | List[str] |
| usernames | List of following usernames | List[str] |
| next_token | Next token for pagination | str |
| data | Complete user data for following | List[Dict[str, True]] |
| includes | Additional data requested via expansions | Dict[str, True] |
| meta | Metadata including pagination info | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Unfollow User

### What it is
This block unfollows a specified Twitter user.

### What it does
This block unfollows a specified Twitter user.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| target_user_id | The user ID of the user that you would like to unfollow | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the unfollow action was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
