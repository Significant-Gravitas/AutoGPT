# Twitter Get Liked Tweets

### What it is
This block gets information about tweets liked by a user.

### What it does
This block gets information about tweets liked by a user.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with your tweets. For example:
- Select 'Media_Keys' to get media details
- Select 'Author_User_ID' to get user information
- Select 'Place_ID' to get location details | ExpansionFilter | No |
| media_fields | Select what media information you want to see (images, videos, etc). To use this, you must first select 'Media_Keys' in the expansions above. | TweetMediaFieldsFilter | No |
| place_fields | Select what location information you want to see (country, coordinates, etc). To use this, you must first select 'Place_ID' in the expansions above. | TweetPlaceFieldsFilter | No |
| poll_fields | Select what poll information you want to see (options, voting status, etc). To use this, you must first select 'Poll_IDs' in the expansions above. | TweetPollFieldsFilter | No |
| tweet_fields | Select what tweet information you want to see. For referenced tweets (like retweets), select 'Referenced_Tweet_ID' in the expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see. To use this, you must first select one of these in expansions above:
- 'Author_User_ID' for tweet authors
- 'Mentioned_Usernames' for mentioned users
- 'Reply_To_User_ID' for users being replied to
- 'Referenced_Tweet_Author_ID' for authors of referenced tweets | TweetUserFieldsFilter | No |
| user_id | ID of the user to get liked tweets for | str | Yes |
| max_results | Maximum number of results to return (5-100) | int | No |
| pagination_token | Token for getting next/previous page of results | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | All Tweet IDs | List[str] |
| texts | All Tweet texts | List[str] |
| userIds | List of user ids that authored the tweets | List[str] |
| userNames | List of user names that authored the tweets | List[str] |
| next_token | Next token for pagination | str |
| data | Complete Tweet data | List[Dict[str, True]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, True] |
| meta | Provides metadata such as pagination info (next_token) or result counts | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Get Liking Users

### What it is
This block gets information about users who liked a tweet.

### What it does
This block gets information about users who liked a tweet.

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
| tweet_id | ID of the tweet to get liking users for | str | Yes |
| max_results | Maximum number of results to return (1-100) | int | No |
| pagination_token | Token for getting next/previous page of results | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | All User IDs who liked the tweet | List[str] |
| username | All User usernames who liked the tweet | List[str] |
| next_token | Next token for pagination | str |
| data | Complete Tweet data | List[Dict[str, True]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, True] |
| meta | Provides metadata such as pagination info (next_token) or result counts | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Like Tweet

### What it is
This block likes a tweet.

### What it does
This block likes a tweet.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| tweet_id | ID of the tweet to like | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the operation was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Unlike Tweet

### What it is
This block unlikes a tweet.

### What it does
This block unlikes a tweet.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| tweet_id | ID of the tweet to unlike | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the operation was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
