# Twitter Like
<!-- MANUAL: file_description -->
Blocks for liking tweets and retrieving liked tweets on Twitter/X.
<!-- END MANUAL -->

## Twitter Get Liked Tweets

### What it is
This block gets information about tweets liked by a user.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve tweets that a specified user has liked. Results are returned in reverse chronological order (most recently liked first) with pagination support.

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
| data | Complete Tweet data | List[Dict[str, Any]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, Any] |
| meta | Provides metadata such as pagination info (next_token) or result counts | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Interest Analysis**: Analyze a user's liked tweets to understand their interests, preferences, and sentiment.

**Content Discovery**: Find high-quality content by examining tweets liked by influencers in your niche.

**Engagement Research**: Study what types of content resonate with your target audience based on their likes.
<!-- END MANUAL -->

---

## Twitter Get Liking Users

### What it is
This block gets information about users who liked a tweet.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve a paginated list of users who have liked a specific tweet. Results include user IDs, usernames, and optionally expanded profile data.

The block uses Tweepy with OAuth 2.0 authentication. Users are returned with pagination support for tweets with many likes. Expansions can include pinned tweet data for each user who liked the tweet.
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
| data | Complete Tweet data | List[Dict[str, Any]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, Any] |
| meta | Provides metadata such as pagination info (next_token) or result counts | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Engagement Analysis**: Identify who is engaging with your content to understand your audience better.

**Influencer Identification**: Find influential users who liked your tweet for potential outreach or collaboration.

**Community Discovery**: Discover potential community members or customers by analyzing who engages with relevant content.
<!-- END MANUAL -->

---

## Twitter Like Tweet

### What it is
This block likes a tweet.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to like a tweet on behalf of the authenticated user. The like is public—the tweet author and others can see that you liked the tweet.

The block authenticates using OAuth 2.0 with like write permissions and sends a POST request to add a like to the specified tweet. Returns a success indicator confirming the like was added.
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
**Engagement Automation**: Like tweets from accounts you want to engage with to increase visibility and interaction.

**Content Appreciation**: Automatically like tweets that mention your brand positively or use specific hashtags.

**Community Building**: Like tweets from community members to acknowledge their contributions and encourage further engagement.
<!-- END MANUAL -->

---

## Twitter Unlike Tweet

### What it is
This block unlikes a tweet.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to remove a like from a tweet. The unlike action is processed silently—the tweet author is not specifically notified that you unliked.

The block authenticates using OAuth 2.0 with like write permissions and sends a DELETE request to remove the like from the specified tweet. Returns a success indicator confirming the like was removed.
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
**Like Cleanup**: Remove likes from tweets you no longer want associated with your account.

**Content Review**: Unlike tweets after reviewing them and determining they don't align with your values.

**Engagement Adjustment**: Adjust your like history as part of curating your public engagement profile.
<!-- END MANUAL -->

---
