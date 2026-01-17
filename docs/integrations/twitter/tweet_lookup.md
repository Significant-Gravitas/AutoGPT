# Twitter Tweet Lookup
<!-- MANUAL: file_description -->
Blocks for retrieving information about specific tweets on Twitter/X.
<!-- END MANUAL -->

## Twitter Get Tweet

### What it is
This block retrieves information about a specific Tweet.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve detailed information about a specific tweet by its ID. Returns tweet content, author information, engagement metrics, and any requested expanded data.

The block uses Tweepy with OAuth 2.0 authentication and supports extensive expansions to include additional data like media, author profile, location, poll results, and referenced tweets. Useful for analyzing individual tweets or verifying tweet content.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with your tweets. For example: - Select 'Media_Keys' to get media details - Select 'Author_User_ID' to get user information - Select 'Place_ID' to get location details | ExpansionFilter | No |
| media_fields | Select what media information you want to see (images, videos, etc). To use this, you must first select 'Media_Keys' in the expansions above. | TweetMediaFieldsFilter | No |
| place_fields | Select what location information you want to see (country, coordinates, etc). To use this, you must first select 'Place_ID' in the expansions above. | TweetPlaceFieldsFilter | No |
| poll_fields | Select what poll information you want to see (options, voting status, etc). To use this, you must first select 'Poll_IDs' in the expansions above. | TweetPollFieldsFilter | No |
| tweet_fields | Select what tweet information you want to see. For referenced tweets (like retweets), select 'Referenced_Tweet_ID' in the expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see. To use this, you must first select one of these in expansions above: - 'Author_User_ID' for tweet authors - 'Mentioned_Usernames' for mentioned users - 'Reply_To_User_ID' for users being replied to - 'Referenced_Tweet_Author_ID' for authors of referenced tweets | TweetUserFieldsFilter | No |
| tweet_id | Unique identifier of the Tweet to request (ex: 1460323737035677698) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | Tweet ID | str |
| text | Tweet text | str |
| userId | ID of the tweet author | str |
| userName | Username of the tweet author | str |
| data | Tweet data | Dict[str, Any] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, Any] |
| meta | Metadata about the tweet | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Content Verification**: Retrieve a specific tweet to verify its content, author, or current engagement metrics.

**Thread Analysis**: Look up individual tweets in a thread to analyze specific parts of a conversation.

**Link Processing**: Fetch tweet details when processing shared Twitter links in your workflows.
<!-- END MANUAL -->

---

## Twitter Get Tweets

### What it is
This block retrieves information about multiple Tweets.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve detailed information about multiple tweets in a single request. Accepts up to 100 tweet IDs and returns comprehensive data for all of them efficiently.

The block uses Tweepy with OAuth 2.0 authentication and supports extensive expansions to include additional data like media, author profiles, and referenced tweets. Returns arrays of tweet IDs, text content, author information, and complete tweet data objects.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with your tweets. For example: - Select 'Media_Keys' to get media details - Select 'Author_User_ID' to get user information - Select 'Place_ID' to get location details | ExpansionFilter | No |
| media_fields | Select what media information you want to see (images, videos, etc). To use this, you must first select 'Media_Keys' in the expansions above. | TweetMediaFieldsFilter | No |
| place_fields | Select what location information you want to see (country, coordinates, etc). To use this, you must first select 'Place_ID' in the expansions above. | TweetPlaceFieldsFilter | No |
| poll_fields | Select what poll information you want to see (options, voting status, etc). To use this, you must first select 'Poll_IDs' in the expansions above. | TweetPollFieldsFilter | No |
| tweet_fields | Select what tweet information you want to see. For referenced tweets (like retweets), select 'Referenced_Tweet_ID' in the expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see. To use this, you must first select one of these in expansions above: - 'Author_User_ID' for tweet authors - 'Mentioned_Usernames' for mentioned users - 'Reply_To_User_ID' for users being replied to - 'Referenced_Tweet_Author_ID' for authors of referenced tweets | TweetUserFieldsFilter | No |
| tweet_ids | List of Tweet IDs to request (up to 100) | List[str] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | All Tweet IDs | List[str] |
| texts | All Tweet texts | List[str] |
| userIds | List of user ids that authored the tweets | List[str] |
| userNames | List of user names that authored the tweets | List[str] |
| data | Complete Tweet data | List[Dict[str, Any]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, Any] |
| meta | Metadata about the tweets | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Batch Processing**: Efficiently retrieve data for multiple tweets at once, such as all tweets in a thread.

**Content Analysis**: Analyze multiple tweets for sentiment, engagement patterns, or content classification.

**Report Generation**: Gather data on multiple tweets for creating engagement reports or content audits.
<!-- END MANUAL -->

---
