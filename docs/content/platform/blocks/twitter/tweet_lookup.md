# Twitter Get Tweet

### What it is
This block retrieves information about a specific Tweet.

### What it does
This block retrieves information about a specific Tweet.

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
| tweet_id | Unique identifier of the Tweet to request (ex: 1460323737035677698) | str | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | Tweet ID | str |
| text | Tweet text | str |
| userId | ID of the tweet author | str |
| userName | Username of the tweet author | str |
| data | Tweet data | Dict[str, True] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, True] |
| meta | Metadata about the tweet | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Twitter Get Tweets

### What it is
This block retrieves information about multiple Tweets.

### What it does
This block retrieves information about multiple Tweets.

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
| tweet_ids | List of Tweet IDs to request (up to 100) | List[str] | Yes |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | All Tweet IDs | List[str] |
| texts | All Tweet texts | List[str] |
| userIds | List of user ids that authored the tweets | List[str] |
| userNames | List of user names that authored the tweets | List[str] |
| data | Complete Tweet data | List[Dict[str, True]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, True] |
| meta | Metadata about the tweets | Dict[str, True] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
