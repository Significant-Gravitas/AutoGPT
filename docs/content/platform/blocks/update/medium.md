

## Get Reddit Posts

### What it is
A specialized block designed to fetch posts from a specified subreddit on Reddit.

### What it does
This block retrieves a collection of posts from a designated subreddit, allowing users to filter posts based on time and limit the number of posts retrieved.

### How it works
The block connects to Reddit using provided credentials, navigates to the specified subreddit, and collects posts based on the given parameters. It can filter posts by time and stop when it reaches a specific post ID.

### Inputs
- Subreddit: The name of the subreddit you want to fetch posts from
- Reddit Credentials: Authentication details needed to access Reddit (client ID, client secret, username, password, and user agent)
- Last Minutes: Optional filter to only get posts from the last X minutes
- Last Post: Optional post ID where the fetching should stop
- Post Limit: Optional limit on the number of posts to fetch (defaults to 10)

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
This block posts a comment on a specified Reddit post using provided credentials and comment content.

### How it works
The block authenticates with Reddit using the provided credentials, locates the specified post using its ID, and posts the given comment content as a reply to that post.

### Inputs
- Reddit Credentials: Authentication details needed to access Reddit (client ID, client secret, username, password, and user agent)
- Comment Data: Contains:
  - Post ID: The ID of the post to comment on
  - Comment: The text content of the comment

### Outputs
- Comment ID: The unique identifier of the posted comment

### Possible use case
An automated customer service system that responds to user inquiries or feedback posted on Reddit, such as providing automated responses to frequently asked questions in a product's subreddit.

