# Twitter API Integration Blocks

## Twitter Post Tweet Block

### What it is
A block that creates tweets on Twitter with various optional attachments and settings.

### What it does
This block allows posting tweets with text content and optional attachments like media, polls, quotes, or deep links.

### How it works
It uses the Twitter API (Tweepy) to create a tweet with the specified content and settings, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_text | Main text content of the tweet |
| attachment | Optional media, deep link, poll, place or quote attachment |
| for_super_followers_only | Whether the tweet is exclusively for super followers |
| exclude_reply_user_ids | User IDs to exclude from reply thread |
| in_reply_to_tweet_id | ID of tweet being replied to |
| reply_settings | Who can reply to the tweet |

### Outputs
| Output | Description |
|--------|-------------|
| tweet_id | ID of the created tweet |
| tweet_url | URL to view the tweet |
| error | Error message if posting failed |

### Possible use case
Automating tweet publishing with rich content like polls, media or quotes.

---

## Twitter Delete Tweet Block

### What it is
A block that deletes a specified tweet on Twitter.

### What it does
This block removes an existing tweet using its tweet ID.

### How it works
It uses the Twitter API (Tweepy) to delete the tweet with the given ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_id | ID of the tweet to delete |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether deletion was successful |
| error | Error message if deletion failed |

### Possible use case
Automated cleanup of old or irrelevant tweets.

---

## Twitter Search Recent Tweets Block

### What it is
A block that searches recent public tweets on Twitter.

### What it does
This block searches for tweets matching specified criteria with options for filtering and pagination.

### How it works
It queries the Twitter API (Tweepy) search endpoint with the provided parameters and returns matching tweets and metadata.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| query | Search query string |
| max_results | Maximum number of results per page |
| pagination | Token for getting next page of results |
| expansions | Additional data fields to include |
| start_time | Start of search time window |
| end_time | End of search time window |
| since_id | Return results after this tweet ID |
| until_id | Return results before this tweet ID |
| sort_order | Order of returned results |

### Outputs
| Output | Description |
|--------|-------------|
| tweet_ids | List of matching tweet IDs |
| tweet_texts | List of tweet text contents |
| next_token | Token for retrieving next page |
| data | Complete tweet data |
| included | Additional requested data |
| meta | Pagination and result metadata |
| error | Error message if search failed |

### Possible use case
Monitoring Twitter for mentions of specific topics or hashtags.

---

## Twitter Get Quote Tweets Block

### What it is
A block that retrieves quote tweets (tweets that quote a specific tweet) from Twitter.

### What it does
This block gets a list of tweets that quote the specified tweet ID, with options for pagination and filtering.

### How it works
It uses the Twitter API (Tweepy) to fetch quote tweets for a given tweet ID, handling authentication and returning tweet data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_id | ID of the tweet to get quotes for |
| max_results | Maximum number of results to return (max 100) |
| exclude | Types of tweets to exclude |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| media_fields | Media-related fields to include [more info](twitter.md#common-input). |
| place_fields | Location-related fields to include [more info](twitter.md#common-input). |
| poll_fields | Poll-related fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of quote tweet IDs |
| texts | List of quote tweet text contents |
| next_token | Token for retrieving next page [more info](twitter.md#common-output). |
| data | Complete tweet data |
| included | Additional requested data [more info](twitter.md#common-output). |
| meta | Pagination and result metadata [more info](twitter.md#common-output). |
| error | Error message if request failed |

### Possible use case
Monitoring engagement and responses to specific tweets through quote tweets.

---

## Twitter Retweet Block

### What it is
A block that retweets an existing tweet on Twitter.

### What it does
This block creates a retweet of the specified tweet using its tweet ID.

### How it works
It uses the Twitter API (Tweepy) to retweet the tweet with the given ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_id | ID of the tweet to retweet |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether retweet was successful |
| error | Error message if retweet failed |

### Possible use case
Automated retweeting of content matching specific criteria.

---

## Twitter Remove Retweet Block

### What it is
A block that removes a retweet on Twitter.

### What it does
This block removes an existing retweet of the specified tweet.

### How it works
It uses the Twitter API (Tweepy) to remove the retweet with the given tweet ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_id | ID of the tweet to remove retweet from |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether retweet removal was successful |
| error | Error message if removal failed |

### Possible use case
Automated cleanup of retweets based on certain conditions.

---

## Twitter Get Retweeters Block

### What it is
A block that retrieves information about users who have retweeted a specific tweet.

### What it does
This block gets a list of users who have retweeted the specified tweet ID, with options for pagination and filtering.

### How it works
It uses the Twitter API (Tweepy) to fetch retweeter information for a given tweet ID, handling authentication and returning user data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_id | ID of the tweet to get retweeters for |
| max_results | Maximum number of results per page (1-100) |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| media_fields | Media-related fields to include [more info](twitter.md#common-input). |
| place_fields | Location-related fields to include [more info](twitter.md#common-input). |
| poll_fields | Poll-related fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of user IDs who retweeted |
| names | List of user names who retweeted |
| usernames | List of usernames who retweeted |
| next_token | Token for retrieving next page |
| data | Complete user data |
| included | Additional requested data [more info](twitter.md#common-output). |
| meta | Pagination and result metadata [more info](twitter.md#common-output). |
| error | Error message if request failed |

### Possible use case
Monitoring engagement and analyzing user behavior through retweet patterns.

---

## Twitter Get User Mentions Block

### What it is
A block that retrieves tweets mentioning a specific Twitter user.

### What it does
This block gets tweets where a user is mentioned, using their user ID.

### How it works
It queries the Twitter API (Tweepy) with the provided user ID to fetch tweets mentioning that user, handling pagination and filters.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| user_id | ID of user to get mentions for |
| max_results | Number of results per page (5-100) |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| media_fields | Media-related fields to include [more info](twitter.md#common-input). |
| place_fields | Location-related fields to include [more info](twitter.md#common-input). |
| poll_fields | Poll-related fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of Tweet IDs |
| texts | List of tweet text contents |
| userIds | List of user IDs who mentioned target user |
| userNames | List of usernames who mentioned target user |
| next_token | Token for retrieving next page |
| data | Complete tweet data |
| included | Additional requested data [more info](twitter.md#common-output). |
| meta | Pagination and result metadata [more info](twitter.md#common-output). |
| error | Error message if request failed |

### Possible use case
Monitoring mentions of specific accounts for community management.

---

## Twitter Get Home Timeline Block

### What it is
A block that retrieves tweets from a user's home timeline.

### What it does
This block returns a collection of recent tweets and retweets posted by the authenticated user and accounts they follow.

### How it works
It uses the Twitter API (Tweepy) to fetch tweets from the home timeline, handling pagination and applying filters.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| max_results | Number of results per page (5-100) |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| media_fields | Media-related fields to include [more info](twitter.md#common-input). |
| place_fields | Location-related fields to include [more info](twitter.md#common-input). |
| poll_fields | Poll-related fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of Tweet IDs |
| texts | List of tweet text contents |
| userIds | List of user IDs who authored tweets |
| userNames | List of usernames who authored tweets |
| next_token | Token for retrieving next page |
| data | Complete tweet data |
| included | Additional requested data [more info](twitter.md#common-output). |
| meta | Pagination and result metadata [more info](twitter.md#common-output). |
| error | Error message if request failed |

### Possible use case
Monitoring and analyzing content from followed accounts.

---

## Twitter Get User Tweets Block

### What it is
A block that retrieves tweets posted by a specific Twitter user.

### What it does
This block returns tweets authored by a single user, identified by their user ID.

### How it works
It uses the Twitter API (Tweepy) to fetch tweets from a specified user's timeline, handling pagination and filters.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| user_id | ID of user to get tweets from |
| max_results | Number of results per page (5-100) |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| media_fields | Media-related fields to include [more info](twitter.md#common-input). |
| place_fields | Location-related fields to include [more info](twitter.md#common-input). |
| poll_fields | Poll-related fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of Tweet IDs |
| texts | List of tweet text contents |
| userIds | List of user IDs who authored tweets |
| userNames | List of usernames who authored tweets |
| next_token | Token for retrieving next page |
| data | Complete tweet data |
| included | Additional requested data [more info](twitter.md#common-output). |
| meta | Pagination and result metadata [more info](twitter.md#common-output). |
| error | Error message if request failed |

### Possible use case
Analyzing content and activity patterns of specific Twitter accounts.

---

## Twitter Get Tweet Block

### What it is
A block that retrieves detailed information about a specific tweet by its ID.

### What it does
This block fetches information about a single tweet specified by the tweet ID, including tweet content, author details, and optional expanded data.

### How it works
It uses the Twitter API (Tweepy) to fetch a single tweet by its ID, handling authentication and returning tweet data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_id | ID of the tweet to fetch |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| media_fields | Media-related fields to include [more info](twitter.md#common-input). |
| place_fields | Location-related fields to include [more info](twitter.md#common-input). |
| poll_fields | Poll-related fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| id | Tweet ID |
| text | Tweet text content |
| userId | ID of tweet author |
| userName | Username of tweet author |
| data | Complete tweet data |
| included | Additional requested data [more info](twitter.md#common-output). |
| meta | Tweet metadata [more info](twitter.md#common-output). |
| error | Error message if request failed |

### Possible use case
Retrieving detailed information about specific tweets for analysis or monitoring.

---

## Twitter Get Tweets Block

### What it is
A block that retrieves information about multiple tweets by their IDs.

### What it does
This block fetches information about multiple tweets (up to 100) specified by their tweet IDs.

### How it works
It uses the Twitter API (Tweepy) to batch fetch tweets by their IDs, handling authentication and returning tweet data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_ids | List of tweet IDs to fetch (max 100) |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| media_fields | Media-related fields to include [more info](twitter.md#common-input). |
| place_fields | Location-related fields to include [more info](twitter.md#common-input). |
| poll_fields | Poll-related fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of tweet IDs |
| texts | List of tweet text contents |
| userIds | List of tweet author IDs |
| userNames | List of tweet author usernames |
| data | Complete tweet data array |
| included | Additional requested data [more info](twitter.md#common-output). |
| meta | Tweet metadata [more info](twitter.md#common-output). |
| error | Error message if request failed |

### Possible use case
Batch retrieval of tweet information for analysis or archival purposes.

---

## Twitter Like Tweet Block

### What it is
A block that likes a tweet on Twitter.

### What it does
This block creates a like on a specified tweet using its tweet ID.

### How it works
It uses the Twitter API (Tweepy) to like the tweet with the given ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_id | ID of the tweet to like |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether like was successful |
| error | Error message if like failed |

### Possible use case
Automated liking of tweets matching specific criteria.

---

## Twitter Get Liking Users Block

### What it is
A block that retrieves information about users who liked a specific tweet.

### What it does
This block gets a list of users who have liked the specified tweet ID, with options for pagination and filtering.

### How it works
It uses the Twitter API (Tweepy) to fetch information about users who liked a given tweet ID, handling authentication and returning user data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_id | ID of tweet to get liking users for |
| max_results | Maximum number of results to return (1-100) |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| id | List of user IDs who liked |
| username | List of usernames who liked |
| next_token | Token for retrieving next page |
| data | Complete user data |
| included | Additional requested data [more info](twitter.md#common-output). |
| meta | Pagination and result metadata [more info](twitter.md#common-output). |
| error | Error message if request failed |

### Possible use case
Analyzing engagement patterns through tracking tweet likes.

---

## Twitter Get Liked Tweets Block

### What it is
A block that retrieves tweets liked by a specific Twitter user.

### What it does
This block gets a list of tweets that have been liked by the specified user ID.

### How it works
It uses the Twitter API (Tweepy) to fetch tweets liked by a given user ID, handling pagination and filters.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| user_id | ID of user to get liked tweets for |
| max_results | Maximum number of results per page (5-100) |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| media_fields | Media-related fields to include [more info](twitter.md#common-input). |
| place_fields | Location-related fields to include [more info](twitter.md#common-input). |
| poll_fields | Poll-related fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of liked tweet IDs |
| texts | List of liked tweet text contents |
| userIds | List of tweet author IDs |
| userNames | List of tweet author usernames |
| next_token | Token for retrieving next page |
| data | Complete tweet data |
| included | Additional requested data [more info](twitter.md#common-output). |
| meta | Pagination and result metadata [more info](twitter.md#common-output). |
| error | Error message if request failed |

### Possible use case
Analyzing user interests and preferences through liked tweet patterns.

---

## Twitter Unlike Tweet Block

### What it is
A block that unlikes a previously liked tweet on Twitter.

### What it does
This block removes a like from the specified tweet using its tweet ID.

### How it works
It uses the Twitter API (Tweepy) to unlike the tweet with the given ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_id | ID of the tweet to unlike |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether unlike was successful |
| error | Error message if unlike failed |

### Possible use case
Automated cleanup of likes based on certain conditions.

---

## Twitter Hide Reply Block

### What it is
A block that hides a reply to one of your tweets.

### What it does
This block hides a specified reply tweet from being visible in the main conversation thread.

### How it works
It uses the Twitter API (Tweepy) to hide a reply tweet with the given tweet ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_id | ID of the tweet reply to hide |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether hiding was successful |
| error | Error message if hiding failed |

### Possible use case
Moderating conversations by hiding inappropriate or unwanted replies.

---

## Twitter Unhide Reply Block

### What it is
A block that unhides a previously hidden reply to a tweet.

### What it does
This block makes a hidden reply tweet visible again in the conversation thread.

### How it works
It uses the Twitter API (Tweepy) to unhide a reply tweet with the given tweet ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_id | ID of the tweet reply to unhide |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether unhiding was successful |
| error | Error message if unhiding failed |

### Possible use case
Restoring previously hidden replies when moderation is no longer needed.

---

## Twitter Bookmark Tweet Block

### What it is
A block that bookmarks a specified tweet on Twitter.

### What it does
This block creates a bookmark for a tweet using its tweet ID.

### How it works
It uses the Twitter API (Tweepy) to bookmark the tweet with the given ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_id | ID of the tweet to bookmark |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether bookmark was successful |
| error | Error message if bookmark failed |

### Possible use case
Saving tweets for later reference and organization.

---

## Twitter Get Bookmarked Tweets Block

### What it is
A block that retrieves a user's bookmarked tweets from Twitter.

### What it does
This block gets a list of tweets that have been bookmarked by the authenticated user.

### How it works
It uses the Twitter API (Tweepy) to fetch bookmarked tweets, handling pagination and optional data expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| max_results | Maximum number of results per page (1-100) |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| media_fields | Media-related fields to include [more info](twitter.md#common-input). |
| place_fields | Location-related fields to include [more info](twitter.md#common-input). |
| poll_fields | Poll-related fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| id | List of bookmarked tweet IDs |
| text | List of bookmarked tweet text contents |
| userId | List of tweet author IDs |
| userName | List of tweet author usernames |
| next_token | Token for retrieving next page |
| data | Complete tweet data |
| included | Additional requested data [more info](twitter.md#common-output). |
| meta | Pagination and result metadata [more info](twitter.md#common-output). |
| error | Error message if request failed |

### Possible use case
Retrieving and analyzing saved tweets for content curation or research.

---

## Twitter Remove Bookmark Tweet Block

### What it is
A block that removes a bookmark from a tweet on Twitter.

### What it does
This block removes an existing bookmark from a specified tweet using its tweet ID.

### How it works
It uses the Twitter API (Tweepy) to remove the bookmark with the given tweet ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| tweet_id | ID of the tweet to remove bookmark from |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether bookmark removal was successful |
| error | Error message if removal failed |

### Possible use case
Managing bookmarks by removing outdated or no longer relevant saved tweets.

---

## Twitter Unblock User Block

### What it is
A block that unblocks a user that has been previously blocked on Twitter.

### What it does
This block removes a block from a specified user using their user ID.

### How it works
It uses the Twitter API (Tweepy) to unblock a user with the given user ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| target_user_id | ID of the user to unblock |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether unblock was successful |
| error | Error message if unblock failed |

### Possible use case
Reverting previously blocked users when access should be restored.

---

## Twitter Get Blocked Users Block

### What it is
A block that retrieves a list of users that have been blocked by the authenticated user.

### What it does
This block gets information about users who have been blocked, with options for pagination and filtering.

### How it works
It uses the Twitter API (Tweepy) to fetch a list of blocked users, handling authentication and returning user data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| max_results | Maximum number of results to return (1-1000) |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include |
| tweet_fields | Tweet-specific fields to include |
| user_fields | User-related fields to include |

### Outputs
| Output | Description |
|--------|-------------|
| user_ids | List of blocked user IDs |
| usernames_ | List of blocked usernames |
| included | Additional requested data |
| meta | Pagination and result metadata |
| next_token | Token for retrieving next page |
| error | Error message if request failed |

### Possible use case
Monitoring and managing blocked users for account safety and moderation.

---

## Twitter Block User Block

### What it is
A block that blocks a user on Twitter.

### What it does
This block blocks a specified user using their user ID.

### How it works
It uses the Twitter API (Tweepy) to block a user with the given user ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| target_user_id | ID of the user to block |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether block was successful |
| error | Error message if block failed |

### Possible use case
Automating user blocking based on specific criteria or behaviors.

## Twitter Unfollow User Block

### What it is
A block that unfollows a Twitter user.

### What it does
This block unfollows a specified user using their user ID.

### How it works
It uses the Twitter API (Tweepy) to unfollow a user with the given user ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| target_user_id | ID of the user to unfollow |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether unfollow was successful |
| error | Error message if unfollow failed |

### Possible use case
Automating unfollowing users based on specific criteria.

---

## Twitter Follow User Block

### What it is
A block that follows a Twitter user.

### What it does
This block follows a specified user using their user ID.

### How it works
It uses the Twitter API (Tweepy) to follow a user with the given user ID, handling authentication and error cases. If the target user has protected tweets, this will send a follow request.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| target_user_id | ID of the user to follow |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether follow was successful |
| error | Error message if follow failed |

### Possible use case
Automating following of users matching specific criteria.

---

## Twitter Get Followers Block

### What it is
A block that retrieves a list of followers for a specified Twitter user.

### What it does
This block gets a list of users who follow the specified user ID, with options for pagination and filtering.

### How it works
It uses the Twitter API (Tweepy) to fetch followers for a given user ID, handling authentication and returning user data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| target_user_id | ID of user to get followers for |
| max_results | Maximum number of results per page (1-1000) |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of follower user IDs |
| usernames | List of follower usernames |
| next_token | Token for retrieving next page |
| data | Complete user data |
| includes | Additional requested data [more info](twitter.md#common-output). |
| meta | Pagination and result metadata [more info](twitter.md#common-output). |
| error | Error message if request failed |

### Possible use case
Analyzing follower patterns and demographics.

---

## Twitter Get Following Block

### What it is
A block that retrieves a list of users that a specified Twitter user follows.

### What it does
This block gets a list of users being followed by the specified user ID, with options for pagination and filtering.

### How it works
It uses the Twitter API (Tweepy) to fetch following list for a given user ID, handling authentication and returning user data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| target_user_id | ID of user to get following list for |
| max_results | Maximum number of results per page (1-1000) |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of following user IDs |
| usernames | List of following usernames |
| next_token | Token for retrieving next page |
| data | Complete user data |
| includes | Additional requested data [more info](twitter.md#common-output). |
| meta | Pagination and result metadata [more info](twitter.md#common-output). |
| error | Error message if request failed |

### Possible use case
Analyzing following patterns and network connections.

---

## Twitter Unmute User Block

### What it is
A block that unmutes a previously muted user on Twitter.

### What it does
This block unmutes a specified user using their user ID. The request succeeds with no action if the target user is not currently muted.

### How it works
It uses the Twitter API (Tweepy) to unmute a user with the given user ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| target_user_id | ID of the user to unmute |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether unmute was successful |
| error | Error message if unmute failed |

### Possible use case
Reverting muted users when communication should be restored.

---

## Twitter Get Muted Users Block

### What it is
A block that retrieves a list of users muted by the authenticated user.

### What it does
This block gets a list of muted users with options for pagination and filtering.

### How it works
It uses the Twitter API (Tweepy) to fetch muted users, handling authentication and returning user data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| max_results | Maximum results per page (1-1000, default 10) |
| pagination_token | Token for getting next/previous page |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of muted user IDs |
| usernames | List of muted usernames |
| next_token | Token for retrieving next page |
| data | Complete user data for muted users |
| includes | Additional requested data |
| meta | Metadata including pagination info |
| error | Error message if request failed |

### Possible use case
Monitoring and managing muted users list for content filtering.

---

## Twitter Mute User Block

### What it is
A block that mutes a specified user on Twitter.

### What it does
This block mutes a user using their user ID to stop seeing their tweets.

### How it works
It uses the Twitter API (Tweepy) to mute a user with the given user ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| target_user_id | ID of the user to mute |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether mute was successful |
| error | Error message if mute failed |

### Possible use case
Automating user muting based on specific criteria or behaviors.

---

## Twitter Get User Block

### What it is
A block that retrieves information about a single Twitter user by either their user ID or username.

### What it does
This block fetches detailed user information, including basic profile data and optional expanded information, for a specified Twitter user.

### How it works
It uses the Twitter API (Tweepy) to fetch user data for a single user identified by either ID or username, handling authentication and returning user data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| identifier | User identifier (either user ID or username) |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| id | User ID |
| username_ | Twitter username |
| name_ | Display name |
| data | Complete user data |
| included | Additional requested data |
| error | Error message if request failed |

### Possible use case
Retrieving detailed user profile information for analysis or verification.

---

## Twitter Get Users Block

### What it is
A block that retrieves information about multiple Twitter users by their IDs or usernames.

### What it does
This block fetches detailed user information for up to 100 users at once, including basic profile data and optional expanded information.

### How it works
It uses the Twitter API (Tweepy) to batch fetch user data for multiple users identified by either IDs or usernames, handling authentication and returning user data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| identifier | List of user identifiers (either user IDs or usernames, max 100) |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of user IDs |
| usernames_ | List of Twitter usernames |
| names_ | List of display names |
| data | Complete user data array |
| included | Additional requested data |
| error | Error message if request failed |

### Possible use case
Batch retrieval of user profile information for analysis or monitoring.

---

## Twitter Search Spaces Block

### What it is
A block that searches for live or scheduled Twitter Spaces by specified search terms.

### What it does
This block searches for Twitter Spaces based on title keywords, with options to filter by state (live/scheduled) and pagination.

### How it works
It uses the Twitter API (Tweepy) to search for Spaces matching the query parameters, handling authentication and returning Space data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| query | Search term to find in Space titles |
| max_results | Maximum number of results to return (1-100, default 10) |
| state | Type of Spaces to return (live, scheduled, or all) |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| space_fields | Space-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of space IDs |
| titles | List of space titles |
| host_ids | List of host IDs |
| next_token | Token for retrieving next page |
| data | Complete space data |
| includes | Additional requested data |
| meta | Metadata including pagination info |
| error | Error message if request failed |

### Possible use case
Finding relevant Twitter Spaces for content discovery and engagement.

---

## Twitter Get Spaces Block

### What it is
A block that retrieves information about multiple Twitter Spaces specified by Space IDs or creator user IDs.

### What it does
This block fetches detailed information for up to 100 Spaces using either their Space IDs or creator user IDs.

### How it works
It uses the Twitter API (Tweepy) to batch fetch Space data for multiple Spaces, handling authentication and returning Space data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| identifier | Choice of lookup by Space IDs or creator user IDs |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| space_fields | Space-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of Space IDs |
| titles | List of Space titles |
| data | Complete Space data array |
| includes | Additional requested data |
| error | Error message if request failed |

### Possible use case
Batch retrieval of Space information for analytics or monitoring.

---

## Twitter Get Space By ID Block

### What it is
A block that retrieves information about a single Twitter Space specified by Space ID.

### What it does
This block fetches detailed information about a single Space, including host information and other metadata.

### How it works
It uses the Twitter API (Tweepy) to fetch Space data for a single Space ID, handling authentication and returning Space data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| space_id | ID of Space to retrieve |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| space_fields | Space-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| id | Space ID |
| title | Space title |
| host_ids | List of host IDs |
| data | Complete Space data |
| includes | Additional requested data |
| error | Error message if request failed |

### Possible use case
Retrieving detailed information about a specific Space for analysis or display.

---

## Twitter Get Space Buyers Block

### What it is
A block that retrieves a list of users who purchased tickets to a Twitter Space.

### What it does
This block gets information about users who bought tickets to attend a specific Space.

### How it works
It uses the Twitter API (Tweepy) to fetch buyer information for a Space, handling authentication and returning user data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| space_id | ID of Space to get buyers for |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| buyer_ids | List of buyer user IDs |
| usernames | List of buyer usernames |
| data | Complete buyer user data |
| includes | Additional requested data |
| error | Error message if request failed |

### Possible use case
Analyzing ticket sales and attendee information for monetized Spaces.

---

## Twitter Get Space Tweets Block

### What it is
A block that retrieves tweets shared in a specific Twitter Space.

### What it does
This block gets tweets that were shared during a Space session.

### How it works
It uses the Twitter API (Tweepy) to fetch tweets from a Space, handling authentication and returning tweet data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| space_id | ID of Space to get tweets for |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| media_fields | Media-related fields to include [more info](twitter.md#common-input). |
| place_fields | Location-related fields to include [more info](twitter.md#common-input). |
| poll_fields | Poll-related fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| tweet_ids | List of tweet IDs |
| texts | List of tweet texts |
| data | Complete tweet data |
| includes | Additional requested data |
| meta | Response metadata |
| error | Error message if request failed |

### Possible use case
Capturing and analyzing content shared during Space sessions.

---

## Twitter Get List Block

### What it is
A block that retrieves detailed information about a specific Twitter List.

### What it does
This block fetches information about a Twitter List specified by its ID, including basic list data and optional expanded information.

### How it works
It uses the Twitter API (Tweepy) to fetch list data for a single list ID, handling authentication and returning list data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| list_id | ID of the Twitter List to retrieve |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| list_fields | List-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| id | List ID |
| name | List name |
| owner_id | ID of List owner |
| owner_username | Username of List owner |
| data | Complete list data |
| included | Additional requested data |
| meta | Response metadata |
| error | Error message if request failed |

### Possible use case
Retrieving detailed information about specific Twitter Lists for analysis or display.

---

## Twitter Get Owned Lists Block

### What it is
A block that retrieves all Twitter Lists owned by a specified user.

### What it does
This block fetches a list of Twitter Lists owned by a user ID, with options for pagination and filtering.

### How it works
It uses the Twitter API (Tweepy) to fetch owned lists for a given user ID, handling authentication and returning list data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| user_id | ID of user whose Lists to retrieve |
| max_results | Maximum results per page (1-100, default 10) |
| pagination_token | Token for getting next page |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| list_fields | List-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| list_ids | List of owned List IDs |
| list_names | List of owned List names |
| next_token | Token for retrieving next page |
| data | Complete List data array |
| included | Additional requested data |
| meta | Metadata including pagination info |
| error | Error message if request failed |

### Possible use case
Analyzing owned Lists for content curation and audience management.

---

## Twitter Remove List Member Block

### What it is
A block that removes a member from a specified Twitter List owned by the authenticated user.

### What it does
This block removes a specified user from a Twitter List they are currently a member of.

### How it works
It uses the Twitter API (Tweepy) to remove a user from a specified List, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| list_id | ID of the List to remove member from |
| user_id | ID of the user to remove from List |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether removal was successful |
| error | Error message if removal failed |

### Possible use case
Managing List membership by removing users who no longer meet List criteria.

---

## Twitter Add List Member Block

### What it is
A block that adds a member to a specified Twitter List owned by the authenticated user.

### What it does
This block adds a specified user as a new member to a Twitter List.

### How it works
It uses the Twitter API (Tweepy) to add a user to a specified List, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| list_id | ID of the List to add member to |
| user_id | ID of the user to add to List |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether addition was successful |
| error | Error message if addition failed |

### Possible use case
Growing List membership by adding users who match List criteria.

---

## Twitter Get List Members Block

### What it is
A block that retrieves all members of a specified Twitter List.

### What it does
This block gets information about users who are members of a given List, with options for pagination and filtering.

### How it works
It uses the Twitter API (Tweepy) to fetch member data for a specified List, handling authentication and returning user data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| list_id | ID of the List to get members from |
| max_results | Maximum results per page (1-100, default 10) |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include |
| tweet_fields | Tweet-related fields to include |
| user_fields | User-related fields to include |

### Outputs
| Output | Description |
|--------|-------------|
| ids | List of member user IDs |
| usernames | List of member usernames |
| next_token | Token for retrieving next page |
| data | Complete user data for members |
| included | Additional requested data |
| meta | Pagination and result metadata |
| error | Error message if request failed |

### Possible use case
Analyzing List membership and member profiles.

---

## Twitter Get List Memberships Block

### What it is
A block that retrieves all Lists that a specified user is a member of.

### What it does
This block gets information about Lists where the specified user is a member, with options for pagination and filtering.

### How it works
It uses the Twitter API (Tweepy) to fetch List membership data for a given user ID, handling authentication and returning List data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| user_id | ID of user to get List memberships for |
| max_results | Maximum results per page (1-100, default 10) |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include |
| list_fields | List-specific fields to include |
| user_fields | User-related fields to include |

### Outputs
| Output | Description |
|--------|-------------|
| list_ids | List of List IDs |
| next_token | Token for retrieving next page |
| data | Complete List membership data |
| included | Additional requested data |
| meta | Metadata about pagination |
| error | Error message if request failed |

### Possible use case
Analyzing a user's List memberships to understand their interests and connections.

---

## Twitter Get List Tweets Block

### What it is
A block that retrieves tweets from a specified Twitter List.

### What it does
This block fetches tweets that have been posted within a given List, with options for pagination, filtering, and expanded data.

### How it works
It uses the Twitter API (Tweepy) to fetch tweets from a specified List, handling authentication and returning tweet data with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| list_id | ID of the List to get tweets from |
| max_results | Maximum number of results per page (1-100, default 10) |
| pagination_token | Token for getting next page of results |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| media_fields | Media-related fields to include [more info](twitter.md#common-input). |
| place_fields | Location-related fields to include [more info](twitter.md#common-input). |
| poll_fields | Poll-related fields to include [more info](twitter.md#common-input). |
| tweet_fields | Tweet-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| tweet_ids | List of tweet IDs from the List |
| texts | List of tweet text contents |
| next_token | Token for retrieving next page |
| data | Complete tweet data array |
| included | Additional requested data [more info](twitter.md#common-output). |
| meta | Pagination and result metadata [more info](twitter.md#common-output). |
| error | Error message if request failed |

### Possible use case
Monitoring and analyzing tweets shared within curated Twitter Lists.

---

## Twitter Delete List Block

### What it is
A block that deletes a Twitter List owned by the authenticated user.

### What it does
This block deletes a specified Twitter List using the List ID.

### How it works
It uses the Twitter API (Tweepy) to delete a specified List, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| list_id | ID of the List to delete |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether deletion was successful |
| error | Error message if deletion failed |

### Possible use case
Removing outdated or unnecessary Twitter Lists.

---

## Twitter Update List Block

### What it is
A block that updates a Twitter List owned by the authenticated user.

### What it does
This block modifies an existing List's name and/or description.

### How it works
It uses the Twitter API (Tweepy) to update List metadata, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| list_id | ID of List to update |
| name | New name for the List (optional) |
| description | New description for the List (optional) |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether update was successful |
| error | Error message if update failed |

### Possible use case
Maintaining List metadata to reflect current purpose or organization.

---

## Twitter Create List Block

### What it is
A block that creates a new Twitter List for the authenticated user.

### What it does
This block creates a new Twitter List with specified name, description and privacy settings.

### How it works
It uses the Twitter API (Tweepy) to create a new List, handling authentication and returning List details.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| name | Name for the new List |
| description | Description of the List (optional) |
| private | Whether List should be private |

### Outputs
| Output | Description |
|--------|-------------|
| url | URL of the created List |
| list_id | ID of the created List |
| error | Error message if creation failed |

### Possible use case
Creating Lists to organize Twitter users around specific topics or interests.

---

## Twitter Unpin List Block

### What it is
A block that allows users to unpin a specified Twitter List.

### What it does
This block removes a Twitter List from the user's pinned Lists.

### How it works
It uses the Twitter API (Tweepy) to unpin a List using its List ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| list_id | ID of the List to unpin |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether unpin was successful |
| error | Error message if unpin failed |

### Possible use case
Managing pinned Lists by removing Lists that are no longer priority.

---

## Twitter Pin List Block

### What it is
A block that allows users to pin a specified Twitter List.

### What it does
This block pins a Twitter List to appear at the top of the user's Lists.

### How it works
It uses the Twitter API (Tweepy) to pin a List using its List ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| list_id | ID of the List to pin |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether pin was successful |
| error | Error message if pin failed |

### Possible use case
Prioritizing important Lists for quick access.

---

## Twitter Get Pinned Lists Block

### What it is
A block that retrieves all Twitter Lists that are pinned by the authenticated user.

### What it does
This block fetches a collection of Lists that have been pinned by the user, with options for additional data and filtering.

### How it works
It uses the Twitter API (Tweepy) to fetch pinned Lists data, handling authentication and returning List information with optional expansions.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| expansions | Additional data fields to include [more info](twitter.md#common-input). |
| list_fields | List-specific fields to include [more info](twitter.md#common-input). |
| user_fields | User-related fields to include [more info](twitter.md#common-input). |

### Outputs
| Output | Description |
|--------|-------------|
| list_ids | List of pinned List IDs |
| list_names | List of pinned List names |
| data | Complete List data |
| included | Additional requested data |
| meta | Response metadata |
| error | Error message if request failed |

### Possible use case
Monitoring and managing pinned Lists for organization and quick access.

---

## Twitter Unfollow List Block

### What it is
A block that unfollows a Twitter List that the authenticated user is currently following.

### What it does
This block unfollows a specified Twitter List using the List ID, removing it from the user's followed Lists.

### How it works
It uses the Twitter API (Tweepy) to unfollow a List with the given List ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| list_id | ID of the List to unfollow |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether unfollow was successful |
| error | Error message if unfollow failed |

### Possible use case
Managing followed Lists by removing Lists that are no longer relevant.

---

## Twitter Follow List Block

### What it is
A block that follows a Twitter List for the authenticated user.

### What it does
This block follows a specified Twitter List using the List ID, adding it to the user's followed Lists.

### How it works
It uses the Twitter API (Tweepy) to follow a List with the given List ID, handling authentication and error cases.

### Inputs
| Input | Description |
|-------|-------------|
| credentials | Twitter API credentials with required scopes |
| list_id | ID of the List to follow |

### Outputs
| Output | Description |
|--------|-------------|
| success | Whether follow was successful |
| error | Error message if follow failed |

### Possible use case
Following Lists that match user interests or contain relevant content.

---
## Common Input

The Twitter API lets you choose what information you want to get back when you make a request. Here are the different types of information you can ask for:

### expansions
Extra information about tweets, pictures, and users that are mentioned or connected

| Field | Description |
|-------|-------------|
| Poll_IDs | Gets information about any polls in the tweet, like voting options and results |
| Media_Keys | Gets details about pictures, videos, or GIFs attached to the tweet |
| Author_User_ID | Gets information about who wrote the tweet, like their profile details |
| Edit_History_Tweet_IDs | Shows if and when the tweet was edited and what changed |
| Mentioned_Usernames | Gets profile information about any @mentioned users |
| Place_ID | Gets details about locations tagged in the tweet |
| Reply_To_User_ID | Gets information about the person this tweet is replying to |
| Referenced_Tweet_ID | Gets details about any tweets this one is quoting or retweeting |
| Referenced_Tweet_Author_ID | Gets profile information about who wrote the original tweets being referenced |

### media_fields
Information about pictures, videos, and other media

| Field | Description |
|-------|-------------|
| Duration_in_Milliseconds | How long a video or audio clip plays for (in milliseconds) |
| Height | How tall the picture or video is in pixels |
| Media_Key | A unique code that identifies this specific piece of media |
| Preview_Image_URL | Web link to a smaller preview version of the picture |
| Media_Type | What kind of media it is (photo, video, GIF, etc.) |
| Media_URL | Web link to view the full media |
| Width | How wide the picture or video is in pixels |
| Public_Metrics | Numbers anyone can see (views, plays, etc.) |
| Non_Public_Metrics | Private numbers only the tweet author can see |
| Organic_Metrics | Numbers about natural engagement (non-promoted) |
| Promoted_Metrics | Numbers about paid promotion performance |
| Alternative_Text | Description of the media for accessibility |
| Media_Variants | Different sizes/qualities available (like HD vs SD video) |

### place_fields
Information about locations mentioned in tweets

| Field | Description |
|-------|-------------|
| Contained_Within_Places | Larger areas this place is part of (like a city within a state) |
| Country | The full country name |
| Country_Code | Short two-letter code for the country (like US for United States) |
| Full_Location_Name | Complete name including city, state, country etc. |
| Geographic_Coordinates | Exact location on a map (latitude and longitude) |
| Place_ID | A unique code that identifies this specific location |
| Place_Name | The main name of the place (like "Times Square") |
| Place_Type | What kind of place it is (city, business, landmark etc.) |

### poll_fields
Information about polls in tweets

| Field | Description |
|-------|-------------|
| Duration_Minutes | How long the poll stays open for voting |
| End_DateTime | The exact date and time when voting closes |
| Poll_ID | A unique code that identifies this specific poll |
| Poll_Options | The different choices people can vote for |
| Voting_Status | Whether voting is still open or closed |

### tweet_fields
Information about the tweets themselves

| Field | Description |
|-------|-------------|
| Tweet_Attachments | All media, links, or polls included in the tweet |
| Author_ID | A unique code identifying who wrote the tweet |
| Context_Annotations | Extra information about what the tweet is about |
| Conversation_ID | Code linking all replies in a conversation |
| Creation_Time | When the tweet was posted |
| Edit_Controls | Whether the tweet can be edited and for how long |
| Tweet_Entities | Special parts of the tweet like #hashtags, @mentions, and links |
| Geographic_Location | Where the tweet was posted from |
| Tweet_ID | A unique code for this specific tweet |
| Reply_To_User_ID | Who this tweet is responding to |
| Language | What language the tweet is written in |
| Public_Metrics | Numbers like retweets, likes, and replies |
| Sensitive_Content_Flag | Warning if tweet might contain sensitive content |
| Referenced_Tweets | Other tweets this one is connected to |
| Reply_Settings | Who is allowed to reply to the tweet |
| Tweet_Source | What app or website was used to post |
| Tweet_Text | The actual words in the tweet |
| Withheld_Content | If the tweet is hidden in certain countries |

### user_fields
Information about Twitter users

| Field | Description |
|-------|-------------|
| Account_Creation_Date | When they joined Twitter |
| User_Bio | The "About me" text on their profile |
| User_Entities | Links and @mentions in their profile |
| User_ID | Their unique Twitter user code |
| User_Location | Where they say they are located |
| Latest_Tweet_ID | Code for their most recent tweet |
| Display_Name | Their full profile name (not @username) |
| Pinned_Tweet_ID | Code for the tweet stuck to top of their profile |
| Profile_Picture_URL | Link to their profile picture |
| Is_Protected_Account | Whether their tweets are private |
| Account_Statistics | Number of followers, following, and tweets |
| Profile_URL | Link to their profile webpage |
| Username | Their @handle they use on Twitter |
| Is_Verified | Whether they have a verification checkmark |
| Verification_Type | What kind of verification they have |
| Content_Withholding_Info | If their content is hidden in certain places |

## Extra notes

- Use combinations of expansions and fields to build precise queries. For instance:
  - To fetch a Tweet with media details, include `expansions=Media_Keys` and relevant `media_fields`.
  - For user data in Tweets, add `expansions=Author_User_ID` and appropriate `user_fields`.

- Data returned under `includes` helps cross-reference expanded data objects with their parent entities using IDs.

## Common Output

The Twitter API returns standardized response elements across many endpoints. Here are the common output fields you'll encounter:

### data
The primary data requested in the response

| Field | Description |
|-------|-------------|
| ID | Unique identifier for the object |
| Type | Type of object (tweet, user, etc) |
| Properties | Object-specific fields like text for tweets |

### includes
Additional expanded data objects referenced in the primary data

| Field | Description |
|-------|-------------|
| Tweets | Full tweet objects that were referenced |
| Users | User profile data for authors/mentions |
| Places | Location data for geo-tagged content |
| Media | Details about attached photos/videos |
| Polls | Information about embedded polls |

### meta
Metadata about the response and pagination

| Field | Description |
|-------|-------------|
| Result_Count | Number of items returned |
| Next_Token | Token to get next page of results |
| Previous_Token | Token to get previous page |
| Newest_ID | Most recent ID in results |
| Oldest_ID | Oldest ID in results |
| Total_Tweet_Count | Total matching tweets (search) |

### errors
Details about any errors that occurred

| Field | Description |
|-------|-------------|
| Title | Brief error description |
| Detail | Detailed error message |
| Type | Error category/classification |
| Status | HTTP status code |

### Non-paginated responses
For single-object lookups:
- data: Contains requested object
- includes: Referenced objects
- errors: Any errors encountered

### Paginated responses
For multi-object lookups:
- data: Array of objects
- includes: Referenced objects
- meta: Pagination details
- errors: Any errors encountered
