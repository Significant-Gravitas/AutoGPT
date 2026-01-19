# Twitter Spaces Lookup
<!-- MANUAL: file_description -->
Blocks for retrieving information about Twitter/X Spaces.
<!-- END MANUAL -->

## Twitter Get Space Buyers

### What it is
This block retrieves a list of users who purchased tickets to a Twitter Space.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve a list of users who purchased tickets to a ticketed Twitter Space. Only the Space creator or hosts can access buyer information.

The block uses Tweepy with OAuth 2.0 authentication and returns buyer user IDs, usernames, and optionally expanded profile data. This is useful for managing ticketed events and understanding your paying audience.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose what extra information you want to get with user data. Currently only 'pinned_tweet_id' is available to see a user's pinned tweet. | UserExpansionsFilter | No |
| tweet_fields | Select what tweet information you want to see in pinned tweets. This only works if you select 'pinned_tweet_id' in expansions above. | TweetFieldsFilter | No |
| user_fields | Select what user information you want to see, like username, bio, profile picture, etc. | TweetUserFieldsFilter | No |
| space_id | Space ID to lookup buyers for | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| buyer_ids | List of buyer IDs | List[str] |
| usernames | List of buyer usernames | List[str] |
| data | Complete space buyers data | List[Dict[str, Any]] |
| includes | Additional data requested via expansions | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Audience Management**: Track who purchased tickets to manage attendee lists and send follow-ups.

**Revenue Tracking**: Monitor ticket buyers for ticketed Spaces to understand revenue and audience composition.

**Exclusive Content Delivery**: Identify ticket buyers to provide exclusive content or resources to paid attendees.
<!-- END MANUAL -->

---

## Twitter Get Space By Id

### What it is
This block retrieves information about a single Twitter Space.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve detailed information about a specific Twitter Space by its ID. Returns Space metadata including title, state, host information, and timing details.

The block uses Tweepy with OAuth 2.0 authentication and supports expansions for additional data about creators, hosts, speakers, and topics. Works for both live and scheduled Spaces.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose additional information you want to get with your Twitter Spaces: - Select 'Invited_Users' to see who was invited - Select 'Speakers' to see who can speak - Select 'Creator' to get details about who made the Space - Select 'Hosts' to see who's hosting - Select 'Topics' to see Space topics | SpaceExpansionsFilter | No |
| space_fields | Choose what Space details you want to see, such as: - Title - Start/End times - Number of participants - Language - State (live/scheduled) - And more | SpaceFieldsFilter | No |
| user_fields | Choose what user information you want to see. This works when you select any of these in expansions above: - 'Creator' for Space creator details - 'Hosts' for host information - 'Speakers' for speaker details - 'Invited_Users' for invited user information | TweetUserFieldsFilter | No |
| space_id | Space ID to lookup | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| id | Space ID | str |
| title | Space title | str |
| host_ids | Host ID | List[str] |
| data | Complete space data | Dict[str, Any] |
| includes | Additional data requested via expansions | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Space Monitoring**: Check the status and details of a specific Space you're interested in or hosting.

**Event Tracking**: Monitor when a scheduled Space goes live or verify its current state.

**Analytics Preparation**: Gather Space metadata before or after an event for reporting and analysis.
<!-- END MANUAL -->

---

## Twitter Get Space Tweets

### What it is
This block retrieves tweets shared in a Twitter Space.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve tweets that were shared during a Twitter Space. This includes tweets pinned or shared by hosts and speakers during the live audio session.

The block uses Tweepy with OAuth 2.0 authentication and supports extensive expansions to include additional data like media, author information, and referenced tweets. Returns tweet IDs, text content, and complete tweet data.
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
| space_id | Space ID to lookup tweets for | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| tweet_ids | List of tweet IDs | List[str] |
| texts | List of tweet texts | List[str] |
| data | Complete space tweets data | List[Dict[str, Any]] |
| includes | Additional data requested via expansions | Dict[str, Any] |
| meta | Response metadata | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Content Curation**: Collect tweets shared during a Space to create summaries or follow-up content.

**Resource Compilation**: Gather links and resources shared during educational or informational Spaces.

**Event Documentation**: Archive tweets from important Spaces for reference or community sharing.
<!-- END MANUAL -->

---

## Twitter Get Spaces

### What it is
This block retrieves information about multiple Twitter Spaces.

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Twitter API v2 to retrieve information about multiple Twitter Spaces in a single request. You can look up Spaces by their IDs or by creator user IDs, making it efficient for batch operations.

The block uses Tweepy with OAuth 2.0 authentication and supports expansions for additional data about creators, hosts, speakers, and topics. Returns arrays of Space IDs, titles, and complete Space data objects.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| expansions | Choose additional information you want to get with your Twitter Spaces: - Select 'Invited_Users' to see who was invited - Select 'Speakers' to see who can speak - Select 'Creator' to get details about who made the Space - Select 'Hosts' to see who's hosting - Select 'Topics' to see Space topics | SpaceExpansionsFilter | No |
| space_fields | Choose what Space details you want to see, such as: - Title - Start/End times - Number of participants - Language - State (live/scheduled) - And more | SpaceFieldsFilter | No |
| user_fields | Choose what user information you want to see. This works when you select any of these in expansions above: - 'Creator' for Space creator details - 'Hosts' for host information - 'Speakers' for speaker details - 'Invited_Users' for invited user information | TweetUserFieldsFilter | No |
| identifier | Choose whether to lookup spaces by their IDs or by creator user IDs | Identifier | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| ids | List of space IDs | List[str] |
| titles | List of space titles | List[str] |
| data | Complete space data | List[Dict[str, Any]] |
| includes | Additional data requested via expansions | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Creator Monitoring**: Track all Spaces hosted by specific creators or influencers you follow.

**Batch Analysis**: Retrieve information about multiple Spaces at once for comparative analysis.

**Schedule Tracking**: Monitor upcoming Spaces from multiple accounts to plan your participation.
<!-- END MANUAL -->

---
