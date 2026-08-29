# Twitter List Tweets Lookup
<!-- MANUAL: file_description -->
Blocks for retrieving tweets from Twitter/X lists.
<!-- END MANUAL -->

## Twitter Get List Tweets

### What it is
This block retrieves tweets from a specified Twitter list.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve tweets posted by members of a specific Twitter List. Results include all tweets from list members in reverse chronological order, providing a curated timeline based on the list's membership.

The block uses Tweepy with OAuth 2.0 authentication and supports extensive expansions to include additional data like media, author profiles, and referenced tweets. Pagination allows retrieving large volumes of list content in batches of up to 100 tweets.
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
| list_id | The ID of the List whose Tweets you would like to retrieve | str | Yes |
| max_results | Maximum number of results per page (1-100) | int | No |
| pagination_token | Token for paginating through results | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| tweet_ids | List of tweet IDs | List[str] |
| texts | List of tweet texts | List[str] |
| next_token | Token for next page of results | str |
| data | Complete list tweets data | List[Dict[str, Any]] |
| included | Additional data requested via expansions | Dict[str, Any] |
| meta | Response metadata including pagination tokens | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Industry News Aggregation**: Retrieve tweets from curated industry expert lists to create automated news digests.

**Topic Monitoring**: Monitor tweets from lists focused on specific topics like AI, crypto, or sports for trend analysis.

**Content Curation**: Pull tweets from influencer lists to identify high-quality content for sharing or engagement.
<!-- END MANUAL -->

---
