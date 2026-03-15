# Twitter Timeline
<!-- MANUAL: file_description -->
Blocks for retrieving Twitter/X timelines and user tweets.
<!-- END MANUAL -->

## Twitter Get Home Timeline

### What it is
This block retrieves the authenticated user's home timeline.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve the authenticated user's home timeline—tweets from accounts they follow and their own tweets. Results are returned in reverse chronological order with optional filtering by time range.

The block uses Tweepy with OAuth 2.0 authentication and supports extensive expansions to include additional data like media, author information, and referenced tweets. Pagination allows retrieving large timelines in batches of up to 100 tweets.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| start_time | Start time in YYYY-MM-DDTHH:mm:ssZ format. If set to a time less than 10 seconds ago, it will be automatically adjusted to 10 seconds ago (Twitter API requirement). | str (date-time) | No |
| end_time | End time in YYYY-MM-DDTHH:mm:ssZ format | str (date-time) | No |
| since_id | Returns results with Tweet ID  greater than this (more recent than), we give priority to since_id over start_time | str | No |
| until_id | Returns results with Tweet ID less than this (that is, older than), and used with since_id | str | No |
| sort_order | Order of returned tweets (recency or relevancy) | str | No |
| expansions | Choose what extra information you want to get with your tweets. For example: - Select 'Media_Keys' to get media details - Select 'Author_User_ID' to get user information - Select 'Place_ID' to get location details | ExpansionFilter | No |
| media_fields | Select what media information you want to see (images, videos, etc). To use this, you must first select 'Media_Keys' in the expansions above. | TweetMediaFieldsFilter | No |
| place_fields | Select what location information you want to see (country, coordinates, etc). To use this, you must first select 'Place_ID' in the expansions above. | TweetPlaceFieldsFilter | No |
| poll_fields | Select what poll information you want to see (options, voting status, etc). To use this, you must first select 'Poll_IDs' in the expansions above. | TweetPollFieldsFilter | No |
| tweet_fields | Select what tweet information you want to see. For referenced tweets (like retweets), select 'Referenced_Tweet_ID' in the expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see. To use this, you must first select one of these in expansions above: - 'Author_User_ID' for tweet authors - 'Mentioned_Usernames' for mentioned users - 'Reply_To_User_ID' for users being replied to - 'Referenced_Tweet_Author_ID' for authors of referenced tweets | TweetUserFieldsFilter | No |
| max_results | Number of tweets to retrieve (5-100) | int | No |
| pagination_token | Token for pagination | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of Tweet IDs | List[str] |
| texts | All Tweet texts | List[str] |
| userIds | List of user ids that authored the tweets | List[str] |
| userNames | List of user names that authored the tweets | List[str] |
| next_token | Next token for pagination | str |
| data | Complete Tweet data | List[Dict[str, Any]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, Any] |
| meta | Provides metadata such as pagination info (next_token) or result counts | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Content Digest**: Create automated summaries of your timeline for daily or weekly review.

**Trend Detection**: Monitor your timeline for emerging topics or conversations among accounts you follow.

**Engagement Automation**: Process timeline content to identify tweets worth engaging with or responding to.
<!-- END MANUAL -->

---

## Twitter Get User Mentions

### What it is
This block retrieves Tweets mentioning a specific user.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve tweets that @mention a specific user. Results include replies to the user's tweets, direct mentions, and tagged responses from other accounts.

The block uses Tweepy with OAuth 2.0 authentication and supports time-based filtering and pagination. Expansions allow including additional data like media, author information, and referenced tweets. Returns tweet IDs, text, author information, and complete tweet data.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| start_time | Start time in YYYY-MM-DDTHH:mm:ssZ format. If set to a time less than 10 seconds ago, it will be automatically adjusted to 10 seconds ago (Twitter API requirement). | str (date-time) | No |
| end_time | End time in YYYY-MM-DDTHH:mm:ssZ format | str (date-time) | No |
| since_id | Returns results with Tweet ID  greater than this (more recent than), we give priority to since_id over start_time | str | No |
| until_id | Returns results with Tweet ID less than this (that is, older than), and used with since_id | str | No |
| sort_order | Order of returned tweets (recency or relevancy) | str | No |
| expansions | Choose what extra information you want to get with your tweets. For example: - Select 'Media_Keys' to get media details - Select 'Author_User_ID' to get user information - Select 'Place_ID' to get location details | ExpansionFilter | No |
| media_fields | Select what media information you want to see (images, videos, etc). To use this, you must first select 'Media_Keys' in the expansions above. | TweetMediaFieldsFilter | No |
| place_fields | Select what location information you want to see (country, coordinates, etc). To use this, you must first select 'Place_ID' in the expansions above. | TweetPlaceFieldsFilter | No |
| poll_fields | Select what poll information you want to see (options, voting status, etc). To use this, you must first select 'Poll_IDs' in the expansions above. | TweetPollFieldsFilter | No |
| tweet_fields | Select what tweet information you want to see. For referenced tweets (like retweets), select 'Referenced_Tweet_ID' in the expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see. To use this, you must first select one of these in expansions above: - 'Author_User_ID' for tweet authors - 'Mentioned_Usernames' for mentioned users - 'Reply_To_User_ID' for users being replied to - 'Referenced_Tweet_Author_ID' for authors of referenced tweets | TweetUserFieldsFilter | No |
| user_id | Unique identifier of the user for whom to return Tweets mentioning the user | str | Yes |
| max_results | Number of tweets to retrieve (5-100) | int | No |
| pagination_token | Token for pagination | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of Tweet IDs | List[str] |
| texts | All Tweet texts | List[str] |
| userIds | List of user ids that mentioned the user | List[str] |
| userNames | List of user names that mentioned the user | List[str] |
| next_token | Next token for pagination | str |
| data | Complete Tweet data | List[Dict[str, Any]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, Any] |
| meta | Provides metadata such as pagination info (next_token) or result counts | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Mention Monitoring**: Track all mentions of your account for customer service or community management.

**Engagement Response**: Identify mentions that require responses or engagement for timely replies.

**Sentiment Analysis**: Analyze mentions to understand how users are talking about you or your brand.
<!-- END MANUAL -->

---

## Twitter Get User Tweets

### What it is
This block retrieves Tweets composed by a single user.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve tweets posted by a specific user. Results include original tweets, replies, and retweets from that user's timeline in reverse chronological order.

The block uses Tweepy with OAuth 2.0 authentication and supports time-based filtering and pagination. Expansions allow including additional data like media, mentioned users, and referenced tweets. Returns tweet IDs, text content, and complete tweet data.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| start_time | Start time in YYYY-MM-DDTHH:mm:ssZ format. If set to a time less than 10 seconds ago, it will be automatically adjusted to 10 seconds ago (Twitter API requirement). | str (date-time) | No |
| end_time | End time in YYYY-MM-DDTHH:mm:ssZ format | str (date-time) | No |
| since_id | Returns results with Tweet ID  greater than this (more recent than), we give priority to since_id over start_time | str | No |
| until_id | Returns results with Tweet ID less than this (that is, older than), and used with since_id | str | No |
| sort_order | Order of returned tweets (recency or relevancy) | str | No |
| expansions | Choose what extra information you want to get with your tweets. For example: - Select 'Media_Keys' to get media details - Select 'Author_User_ID' to get user information - Select 'Place_ID' to get location details | ExpansionFilter | No |
| media_fields | Select what media information you want to see (images, videos, etc). To use this, you must first select 'Media_Keys' in the expansions above. | TweetMediaFieldsFilter | No |
| place_fields | Select what location information you want to see (country, coordinates, etc). To use this, you must first select 'Place_ID' in the expansions above. | TweetPlaceFieldsFilter | No |
| poll_fields | Select what poll information you want to see (options, voting status, etc). To use this, you must first select 'Poll_IDs' in the expansions above. | TweetPollFieldsFilter | No |
| tweet_fields | Select what tweet information you want to see. For referenced tweets (like retweets), select 'Referenced_Tweet_ID' in the expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see. To use this, you must first select one of these in expansions above: - 'Author_User_ID' for tweet authors - 'Mentioned_Usernames' for mentioned users - 'Reply_To_User_ID' for users being replied to - 'Referenced_Tweet_Author_ID' for authors of referenced tweets | TweetUserFieldsFilter | No |
| user_id | Unique identifier of the Twitter account (user ID) for whom to return results | str | Yes |
| max_results | Number of tweets to retrieve (5-100) | int | No |
| pagination_token | Token for pagination | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of Tweet IDs | List[str] |
| texts | All Tweet texts | List[str] |
| userIds | List of user ids that authored the tweets | List[str] |
| userNames | List of user names that authored the tweets | List[str] |
| next_token | Next token for pagination | str |
| data | Complete Tweet data | List[Dict[str, Any]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, Any] |
| meta | Provides metadata such as pagination info (next_token) or result counts | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Competitor Monitoring**: Track tweets from competitor accounts to understand their messaging and strategy.

**Content Archiving**: Archive tweets from specific accounts for research or compliance purposes.

**Influencer Analysis**: Analyze posting patterns and content from influencers in your industry.
<!-- END MANUAL -->

---
