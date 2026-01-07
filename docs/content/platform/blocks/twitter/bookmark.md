# Twitter Bookmark Tweet

### What it is
This block bookmarks a tweet on Twitter.

### What it does
This block bookmarks a tweet on Twitter.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| tweet_id | ID of the tweet to bookmark | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the bookmark was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Get Bookmarked Tweets

### What it is
This block retrieves bookmarked tweets from Twitter.

### What it does
This block retrieves bookmarked tweets from Twitter.

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
| max_results | Maximum number of results to return (1-100) | int | No |
| pagination_token | Token for pagination | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | All Tweet IDs | List[str] |
| text | All Tweet texts | List[str] |
| userId | IDs of the tweet authors | List[str] |
| userName | Usernames of the tweet authors | List[str] |
| data | Complete Tweet data | List[Dict[str, True]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, True] |
| meta | Provides metadata such as pagination info (next_token) or result counts | Dict[str, True] |
| next_token | Next token for pagination | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Remove Bookmark Tweet

### What it is
This block removes a bookmark from a tweet on Twitter.

### What it does
This block removes a bookmark from a tweet on Twitter.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| tweet_id | ID of the tweet to remove bookmark from | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the bookmark removal failed | str |
| success | Whether the bookmark was successfully removed | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
