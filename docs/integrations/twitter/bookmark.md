# Twitter Bookmark
<!-- MANUAL: file_description -->
Blocks for managing Twitter/X bookmarks.
<!-- END MANUAL -->

## Twitter Bookmark Tweet

### What it is
This block bookmarks a tweet on Twitter.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to add a tweet to the authenticated user's bookmarks. The bookmark is private and only visible to you—the tweet author is not notified.

The block authenticates using OAuth 2.0 with bookmark write permissions and sends a POST request to add the specified tweet ID to your bookmarks. Returns a success indicator confirming the bookmark was added.
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
**Content Curation**: Save interesting tweets for later reading or to compile into a newsletter or blog post.

**Research Collection**: Bookmark tweets containing valuable information or sources for ongoing research projects.

**Reference Library**: Build a collection of useful tips, tutorials, or resource links shared on Twitter.
<!-- END MANUAL -->

---

## Twitter Get Bookmarked Tweets

### What it is
This block retrieves bookmarked tweets from Twitter.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve tweets that the authenticated user has bookmarked. Results are returned in reverse chronological order (most recently bookmarked first) with pagination support.

The block uses Tweepy with OAuth 2.0 authentication and supports extensive expansions to include additional data like media, author information, and location details. Returns tweet IDs, text content, author information, and complete tweet data objects.
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
| data | Complete Tweet data | List[Dict[str, Any]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, Any] |
| meta | Provides metadata such as pagination info (next_token) or result counts | Dict[str, Any] |
| next_token | Next token for pagination | str |

### Possible use case
<!-- MANUAL: use_case -->
**Content Review**: Process your bookmarked tweets to extract and organize information you've saved.

**Bookmark Cleanup**: Review and categorize bookmarks to identify content to keep, share, or remove.

**Reading List Management**: Retrieve bookmarked tweets to create a structured reading list or export to another system.
<!-- END MANUAL -->

---

## Twitter Remove Bookmark Tweet

### What it is
This block removes a bookmark from a tweet on Twitter.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to remove a tweet from the authenticated user's bookmarks. The operation is private—no one else is notified that you unbookmarked the tweet.

The block authenticates using OAuth 2.0 with bookmark write permissions and sends a DELETE request to remove the specified tweet ID from your bookmarks. Returns a success indicator confirming the bookmark was removed.
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
**Bookmark Cleanup**: Remove tweets you've already read or that are no longer relevant from your bookmarks.

**Content Processing**: Automatically remove bookmarks after extracting or processing the content.

**List Management**: Maintain a manageable bookmark collection by removing older or processed items.
<!-- END MANUAL -->

---
