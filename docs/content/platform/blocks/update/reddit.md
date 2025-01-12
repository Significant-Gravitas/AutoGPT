
## Get Reddit Posts

### What it is
A block designed to fetch posts from a specified subreddit on Reddit.

### What it does
This block retrieves a collection of posts from a designated subreddit, allowing users to specify various filtering criteria such as time limits and post limits.

### How it works
The block connects to Reddit using provided credentials, accesses the specified subreddit, and retrieves posts based on the given parameters. It can filter posts based on time and stop when reaching a specific post ID.

### Inputs
- Subreddit: The name of the subreddit from which to fetch posts
- Credentials: Reddit authentication details including client ID, client secret, username, password, and user agent
- Last Minutes: Optional filter to only fetch posts from the last X minutes
- Last Post: Optional post ID where the fetching should stop
- Post Limit: Maximum number of posts to fetch (defaults to 10)

### Outputs
- Post: A Reddit post object containing:
  - ID: Unique identifier of the post
  - Subreddit: Name of the subreddit
  - Title: Post title
  - Body: Post content

### Possible use case
A social media monitoring tool that needs to track recent discussions in specific Reddit communities, such as gathering customer feedback from a company's subreddit.

## Post Reddit Comment

### What it is
A block that enables posting comments on Reddit posts.

### What it does
This block takes a comment and posts it as a reply to a specified Reddit post.

### How it works
The block authenticates with Reddit using provided credentials, locates the specified post using its ID, and adds the provided comment as a reply to that post.

### Inputs
- Credentials: Reddit authentication details including client ID, client secret, username, password, and user agent
- Comment Data: Contains:
  - Post ID: The ID of the post to comment on
  - Comment: The text content of the comment

### Outputs
- Comment ID: The unique identifier of the posted comment

### Possible use case
An automated customer service system that needs to respond to customer inquiries or posts on a company's Reddit community, or a social media management tool that schedules and posts responses to Reddit discussions.
