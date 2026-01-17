# Twitter Quote
<!-- MANUAL: file_description -->
Blocks for retrieving quote tweets on Twitter/X.
<!-- END MANUAL -->

## Twitter Get Quote Tweets

### What it is
This block gets quote tweets for a specific tweet.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve tweets that quote a specific tweet. Quote tweets are retweets with added commentary, allowing users to share the original tweet while adding their own thoughts.

The block uses Tweepy with OAuth 2.0 authentication and supports extensive expansions to include additional data like media, author information, and location details. Returns paginated results with tweet IDs, text content, and complete tweet data.
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
| tweet_id | ID of the tweet to get quotes for | str | Yes |
| max_results | Number of results to return (max 100) | int | No |
| exclude | Types of tweets to exclude | TweetExcludesFilter | No |
| pagination_token | Token for pagination | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | All Tweet IDs  | List[Any] |
| texts | All Tweet texts | List[Any] |
| next_token | Next token for pagination | str |
| data | Complete Tweet data | List[Dict[str, Any]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, Any] |
| meta | Provides metadata such as pagination info (next_token) or result counts | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Sentiment Analysis**: Analyze how users are commenting on your tweets through quote tweets to understand sentiment.

**Engagement Monitoring**: Track quote tweets to identify discussions and conversations sparked by your content.

**Influencer Discovery**: Find users who quote-tweet your content to identify potential advocates or collaborators.
<!-- END MANUAL -->

---
