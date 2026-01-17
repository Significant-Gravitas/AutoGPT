# Twitter Retweet
<!-- MANUAL: file_description -->
Blocks for retweeting and getting retweet information on Twitter/X.
<!-- END MANUAL -->

## Twitter Get Retweeters

### What it is
This block gets information about who has retweeted a tweet.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve a paginated list of users who have retweeted a specific tweet. Results include user IDs, usernames, display names, and optionally expanded profile data.

The block uses Tweepy with OAuth 2.0 authentication. Users are returned with pagination support for tweets with many retweets. Expansions can include pinned tweet data for each user who retweeted.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with user data. Currently only 'pinned_tweet_id' is available to see a user's pinned tweet. | UserExpansionsFilter | No |
| tweet_fields | Select what tweet information you want to see in pinned tweets. This only works if you select 'pinned_tweet_id' in expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see, like username, bio, profile picture, etc. | TweetUserFieldsFilter | No |
| tweet_id | ID of the tweet to get retweeters for | str | Yes |
| max_results | Maximum number of results per page (1-100) | int | No |
| pagination_token | Token for pagination | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of user ids who retweeted | List[Any] |
| names | List of user names who retweeted | List[Any] |
| usernames | List of user usernames who retweeted | List[Any] |
| next_token | Token for next page of results | str |
| data | Complete Tweet data | List[Dict[str, Any]] |
| included | Additional data that you have requested (Optional) via Expansions field | Dict[str, Any] |
| meta | Provides metadata such as pagination info (next_token) or result counts | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Reach Analysis**: Identify who is amplifying your content to understand your audience's network.

**Influencer Identification**: Find influential users who retweet your content for potential partnerships.

**Campaign Tracking**: Monitor retweets during campaigns to measure virality and engagement.
<!-- END MANUAL -->

---

## Twitter Remove Retweet

### What it is
This block removes a retweet on Twitter.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to remove a retweet from the authenticated user's account. The original tweet remains unaffected—only your retweet is removed.

The block authenticates using OAuth 2.0 with tweet write permissions and sends a DELETE request to undo the retweet. Returns a success indicator confirming the retweet was removed.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| tweet_id | ID of the tweet to remove retweet | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the retweet was successfully removed | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Content Curation**: Remove retweets that are no longer relevant or that you no longer want to amplify.

**Account Cleanup**: Undo accidental retweets or clean up your timeline.

**Brand Alignment**: Remove retweets of content that no longer aligns with your messaging or values.
<!-- END MANUAL -->

---

## Twitter Retweet

### What it is
This block retweets a tweet on Twitter.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Twitter API v2 via Tweepy to retweet a tweet on behalf of the authenticated user. The retweet shares the original tweet with your followers without adding commentary.

The block authenticates using OAuth 2.0 with tweet write permissions and sends a POST request to create the retweet. Returns a success indicator confirming the retweet was created.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| tweet_id | ID of the tweet to retweet | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the retweet was successful | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Content Amplification**: Automatically retweet content from partner accounts or industry news sources.

**Community Engagement**: Retweet positive mentions or user-generated content to show appreciation.

**Information Sharing**: Amplify important announcements or breaking news to your followers.
<!-- END MANUAL -->

---
