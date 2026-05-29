# Twitter Manage
<!-- MANUAL: file_description -->
Blocks for posting, deleting, and managing tweets on Twitter/X.
<!-- END MANUAL -->

## Twitter Delete Tweet

### What it is
This block deletes a tweet on Twitter.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to permanently delete a tweet from the authenticated user's account. The deletion is immediate and cannot be undone.

The block authenticates using OAuth 2.0 with tweet write permissions and sends a DELETE request for the specified tweet ID. Only the tweet's author can delete their own tweets. Returns a success indicator confirming the deletion.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| tweet_id | ID of the tweet to delete | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the tweet deletion failed | str |
| success | Whether the tweet was successfully deleted | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Content Cleanup**: Remove outdated tweets that contain incorrect information or expired promotions.

**Error Correction**: Delete tweets with typos or mistakes before reposting corrected versions.

**Reputation Management**: Remove tweets that may no longer align with your brand messaging or values.
<!-- END MANUAL -->

---

## Twitter Post Tweet

### What it is
This block posts a tweet on Twitter.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to publish a new tweet from the authenticated user's account. The tweet can include text, media attachments, polls, location tags, and quote references.

The block supports advanced features like restricting replies to mentioned users or followers only, posting exclusively for Super Followers, and threading by replying to existing tweets. Returns the created tweet's ID and URL on success.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| tweet_text | Text of the tweet to post | str | No |
| for_super_followers_only | Tweet exclusively for Super Followers | bool | No |
| attachment | Additional tweet data (media, deep link, poll, place or quote) | Any | No |
| exclude_reply_user_ids | User IDs to exclude from reply Tweet thread. [ex - 6253282] | List[str] | No |
| in_reply_to_tweet_id | Tweet ID being replied to. Please note that in_reply_to_tweet_id needs to be in the request if exclude_reply_user_ids is present | str | No |
| reply_settings | Who can reply to the Tweet (mentionedUsers or following) | TweetReplySettingsFilter | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the tweet posting failed | str |
| tweet_id | ID of the created tweet | str |
| tweet_url | URL to the tweet | str |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Publishing**: Schedule and post tweets automatically based on content calendars or triggers.

**Bot Responses**: Create bots that post helpful information or automated responses.

**Content Distribution**: Automatically share new blog posts, product updates, or announcements on Twitter.
<!-- END MANUAL -->

---

## Twitter Search Recent Tweets

### What it is
This block searches all public Tweets in Twitter history.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to search for public tweets matching a query. The search supports Twitter's advanced query syntax including boolean operators, hashtags, mentions, and filters for engagement metrics.

Results can be filtered by time range (start/end times), sorted by recency or relevance, and paginated for retrieving large result sets. The block supports expansions to include additional data like media, author information, and referenced tweets.
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
| query | Search query (up to 1024 characters) | str | Yes |
| max_results | Maximum number of results per page (10-500) | int | No |
| pagination | Token for pagination | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| tweet_ids | All Tweet IDs | List[str] |
| tweet_texts | All Tweet texts | List[str] |
| next_token | Next token for pagination | str |
| data | Complete Tweet data | List[Dict[str, Any]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, Any] |
| meta | Provides metadata such as pagination info (next_token) or result counts | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Brand Monitoring**: Search for mentions of your brand, products, or key terms to monitor sentiment and engagement.

**Competitor Analysis**: Track competitor mentions and campaigns to understand market positioning.

**Trend Research**: Search for trending topics or hashtags to identify opportunities for engagement or content creation.
<!-- END MANUAL -->

---
