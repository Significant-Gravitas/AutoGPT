# Reddit Interaction Blocks

## Get Reddit Posts

### What it is
A block that retrieves posts from a specified subreddit on Reddit.

### What it does
This block fetches a set number of recent posts from a given subreddit, allowing users to collect content from Reddit for various purposes.

### How it works
The block connects to Reddit using provided credentials, accesses the specified subreddit, and retrieves posts based on the given parameters. It can limit the number of posts, stop at a specific post, or fetch posts within a certain time frame.

### Inputs
| Input | Description |
|-------|-------------|
| Subreddit | The name of the subreddit to fetch posts from |
| Reddit Credentials | Login information for accessing Reddit |
| Last Minutes | An optional time limit to stop fetching posts (in minutes) |
| Last Post | An optional post ID to stop fetching when reached |
| Post Limit | The maximum number of posts to fetch (default is 10) |

### Outputs
| Output | Description |
|--------|-------------|
| Post | A Reddit post containing the post ID, subreddit name, title, and body text |

### Possible use case
A content curator could use this block to gather recent posts from a specific subreddit for analysis, summarization, or inclusion in a newsletter.

---

## Post Reddit Comment

### What it is
A block that posts a comment on a specified Reddit post.

### What it does
This block allows users to submit a comment to a particular Reddit post using provided credentials and comment data.

### How it works
The block connects to Reddit using the provided credentials, locates the specified post, and then adds the given comment to that post.

### Inputs
| Input | Description |
|-------|-------------|
| Reddit Credentials | Login information for accessing Reddit |
| Comment Data | Contains the post ID to comment on and the comment text |

### Outputs
| Output | Description |
|--------|-------------|
| Comment ID | The unique identifier of the newly posted comment |

### Possible use case
An automated moderation system could use this block to post pre-defined responses or warnings on Reddit posts that violate community guidelines.