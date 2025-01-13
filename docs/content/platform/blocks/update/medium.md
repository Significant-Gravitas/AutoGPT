
<file_name>autogpt_platform/backend/backend/blocks/reddit.md</file_name>

# Reddit Blocks Documentation

## Get Reddit Posts

### What it is
A block that retrieves posts from a specified subreddit on Reddit.

### What it does
This block fetches a collection of posts from a designated subreddit, allowing you to filter posts based on time and limit the number of posts retrieved.

### How it works
The block connects to Reddit using provided credentials, accesses the specified subreddit, and retrieves posts based on the given parameters. It can filter posts by time and stop when reaching a specific post ID.

### Inputs
- Subreddit: The name of the subreddit you want to fetch posts from
- Reddit Credentials: Authentication details required to access Reddit (client ID, client secret, username, password, and user agent)
- Last Minutes: Optional filter to only fetch posts from the last X minutes
- Last Post: Optional post ID where the fetching should stop
- Post Limit: Maximum number of posts to retrieve (defaults to 10)

### Outputs
- Post: A Reddit post containing:
  - ID: Unique identifier of the post
  - Subreddit: Name of the subreddit
  - Title: Post title
  - Body: Post content

### Possible use case
Monitoring a specific subreddit for new posts about a particular topic, such as tracking customer feedback on a company's subreddit.

## Post Reddit Comment

### What it is
A block that posts comments on Reddit posts.

### What it does
This block allows you to submit comments on specific Reddit posts using provided credentials.

### How it works
The block authenticates with Reddit using the provided credentials, locates the specified post using its ID, and posts a comment on that post.

### Inputs
- Reddit Credentials: Authentication details required to access Reddit (client ID, client secret, username, password, and user agent)
- Comment Data:
  - Post ID: The ID of the post to comment on
  - Comment: The text content of the comment

### Outputs
- Comment ID: The unique identifier of the posted comment

### Possible use case
Automatically responding to customer inquiries or providing automated updates on specific Reddit posts, such as posting status updates on a company's announcements.

