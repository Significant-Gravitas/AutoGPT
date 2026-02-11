# Misc
<!-- MANUAL: file_description -->
Miscellaneous blocks including agent execution, scheduling, HTTP requests, webhooks, and other utility functions.
<!-- END MANUAL -->

## Agent Executor

### What it is
Executes an existing agent inside your agent

### How it works
<!-- MANUAL: how_it_works -->
This block runs another agent as a sub-agent within your workflow. You provide the agent's graph ID, version, and input data, and the block executes that agent and returns its outputs.

Input and output schemas define the expected data structure for communication between the parent and child agents, enabling modular, reusable agent composition.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| user_id | User ID | str | Yes |
| graph_id | Graph ID | str | Yes |
| graph_version | Graph Version | int | Yes |
| agent_name | Name to display in the Builder UI | str | No |
| inputs | Input data for the graph | Dict[str, Any] | Yes |
| input_schema | Input schema for the graph | Dict[str, Any] | Yes |
| output_schema | Output schema for the graph | Dict[str, Any] | Yes |

### Possible use case
<!-- MANUAL: use_case -->
**Modular Workflows**: Break complex workflows into smaller, reusable agents that can be composed together.

**Specialized Agents**: Call domain-specific agents (like a research agent or formatter) from a main orchestration agent.

**Dynamic Routing**: Execute different agents based on input type or user preferences.
<!-- END MANUAL -->

---

## Create Reddit Post

### What it is
Create a new post on a subreddit. Can create text posts or link posts.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to create a new post in the specified subreddit. Provide the title and either text content for a self-post or a URL for a link post. Optionally apply flair using a flair ID from the GetSubredditFlairsBlock.

The block returns the created post's ID and URL, which can be used for chaining with comment blocks or monitoring.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| subreddit | Subreddit to post to, excluding the /r/ prefix | str | Yes |
| title | Title of the post | str | Yes |
| content | Body text of the post (for text posts) | str | No |
| url | URL to submit (for link posts). If provided, content is ignored. | str | No |
| flair_id | Flair template ID to apply to the post (from GetSubredditFlairsBlock) | str | No |
| flair_text | Custom flair text (only used if the flair template allows editing) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_id | ID of the created post | str |
| post_url | URL of the created post | str |
| subreddit | The subreddit name (pass-through for chaining) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Content Distribution**: Automatically share articles or content to relevant subreddits.

**Community Engagement**: Post updates or announcements to subreddit communities.

**Automated Posting**: Schedule and post content to Reddit based on workflow triggers.
<!-- END MANUAL -->

---

## Delete Reddit Comment

### What it is
Delete a Reddit comment that you own.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to delete a comment you previously posted. The deletion is permanent and removes the comment from the post thread. You can only delete your own comments.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| comment_id | The ID of the comment to delete (must be your own comment) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if deletion failed | str |
| success | Whether the deletion was successful | bool |
| comment_id | The comment ID (pass-through for chaining) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Content Cleanup**: Remove outdated or incorrect comments from discussions.

**Automated Moderation**: Delete comments that fail quality checks or receive negative feedback.
<!-- END MANUAL -->

---

## Delete Reddit Post

### What it is
Delete a Reddit post that you own.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to delete a post you previously created. The deletion is permanent and removes the post from the subreddit. You can only delete your own posts.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post_id | The ID of the post to delete (must be your own post) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if deletion failed | str |
| success | Whether the deletion was successful | bool |
| post_id | The post ID (pass-through for chaining) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Content Management**: Remove posts that are no longer relevant or contain errors.

**Automated Cleanup**: Delete posts based on performance metrics or time-based rules.
<!-- END MANUAL -->

---

## Edit Reddit Post

### What it is
Edit the body text of an existing Reddit post that you own. Only works for self/text posts.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to edit the body text of a self-post you created. Link posts cannot be edited. The new content replaces the existing post body.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post_id | The ID of the post to edit (must be your own post) | str | Yes |
| new_content | The new body text for the post | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the edit failed | str |
| success | Whether the edit was successful | bool |
| post_id | The post ID (pass-through for chaining) | str |
| post_url | URL of the edited post | str |

### Possible use case
<!-- MANUAL: use_case -->
**Content Updates**: Update posts with new information or corrections.

**Dynamic Content**: Modify post content based on changing data or feedback.
<!-- END MANUAL -->

---

## Execute Code

### What it is
Executes code in a sandbox environment with internet access.

### How it works
<!-- MANUAL: how_it_works -->
This block executes Python, JavaScript, or Bash code in an isolated E2B sandbox with internet access. Use setup_commands to install dependencies before running your code.

The sandbox includes pip and npm pre-installed. Set timeout to limit execution time, and use dispose_sandbox to clean up after execution or keep the sandbox running for follow-up steps.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| setup_commands | Shell commands to set up the sandbox before running the code. You can use `curl` or `git` to install your desired Debian based package manager. `pip` and `npm` are pre-installed.  These commands are executed with `sh`, in the foreground. | List[str] | No |
| code | Code to execute in the sandbox | str | No |
| language | Programming language to execute | "python" \| "js" \| "bash" \| "r" \| "java" | No |
| timeout | Execution timeout in seconds | int | No |
| dispose_sandbox | Whether to dispose of the sandbox immediately after execution. If disabled, the sandbox will run until its timeout expires. | bool | No |
| template_id | You can use an E2B sandbox template by entering its ID here. Check out the E2B docs for more details: [E2B - Sandbox template](https://e2b.dev/docs/sandbox-template) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| main_result | The main result from the code execution | Main Result |
| results | List of results from the code execution | List[CodeExecutionResult] |
| response | Text output (if any) of the main execution result | str |
| stdout_logs | Standard output logs from execution | str |
| stderr_logs | Standard error logs from execution | str |
| files | Files created or modified during execution. Each file has path, name, content, and workspace_ref (if stored). | List[SandboxFileOutput] |

### Possible use case
<!-- MANUAL: use_case -->
**Data Processing**: Run Python scripts to transform, analyze, or visualize data that can't be handled by standard blocks.

**Custom Integrations**: Execute code to call APIs or services not covered by built-in blocks.

**Dynamic Computation**: Generate and execute code based on AI suggestions for flexible problem-solving.
<!-- END MANUAL -->

---

## Execute Code Step

### What it is
Execute code in a previously instantiated sandbox.

### How it works
<!-- MANUAL: how_it_works -->
This block executes additional code in a sandbox that was previously created with the Instantiate Code Sandbox block. The sandbox maintains state between steps, so variables and installed packages persist.

Use this for multi-step code execution where each step builds on previous results. Set dispose_sandbox to true on the final step to clean up.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| sandbox_id | ID of the sandbox instance to execute the code in | str | Yes |
| step_code | Code to execute in the sandbox | str | No |
| language | Programming language to execute | "python" \| "js" \| "bash" \| "r" \| "java" | No |
| dispose_sandbox | Whether to dispose of the sandbox after executing this code. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| main_result | The main result from the code execution | Main Result |
| results | List of results from the code execution | List[CodeExecutionResult] |
| response | Text output (if any) of the main execution result | str |
| stdout_logs | Standard output logs from execution | str |
| stderr_logs | Standard error logs from execution | str |

### Possible use case
<!-- MANUAL: use_case -->
**Iterative Processing**: Load data in one step, transform it in another, and export in a third.

**Stateful Computation**: Build up results across multiple code executions with shared variables.

**Interactive Analysis**: Run exploratory data analysis steps sequentially in the same environment.
<!-- END MANUAL -->

---

## Get Reddit Comment

### What it is
Get details about a specific Reddit comment by its ID.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to retrieve detailed information about a specific comment by its ID. Returns the comment content, author, score, timestamp, and other metadata.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| comment_id | The ID of the comment to fetch | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if comment couldn't be fetched | str |
| comment | The comment details | RedditComment |

### Possible use case
<!-- MANUAL: use_case -->
**Comment Analysis**: Analyze specific comments for sentiment or content moderation.

**Thread Tracking**: Monitor specific comments for engagement or replies.
<!-- END MANUAL -->

---

## Get Reddit Comment Replies

### What it is
Get replies to a specific Reddit comment.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to fetch replies to a specific comment. Returns a list of direct replies with their content, authors, and metadata.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| comment_id | The ID of the comment to get replies from | str | Yes |
| post_id | The ID of the post containing the comment | str | Yes |
| limit | Maximum number of replies to fetch (max 50) | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if replies couldn't be fetched | str |
| reply | A reply to the comment | RedditComment |
| replies | All replies | List[RedditComment] |
| comment_id | The parent comment ID (pass-through for chaining) | str |
| post_id | The post ID (pass-through for chaining) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Conversation Threading**: Build complete comment threads for analysis or display.

**Response Monitoring**: Track replies to your comments for engagement purposes.
<!-- END MANUAL -->

---

## Get Reddit Inbox

### What it is
Get messages, mentions, and comment replies from your Reddit inbox.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to fetch items from your Reddit inbox. Filter by type to get all items, unread only, direct messages, username mentions, or replies to your comments.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| inbox_type | Type of inbox items to fetch | "all" \| "unread" \| "messages" \| "mentions" \| "comment_replies" | No |
| limit | Maximum number of items to fetch | int | No |
| mark_read | Whether to mark fetched items as read | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if fetch failed | str |
| item | An inbox item | RedditInboxItem |
| items | All fetched items | List[RedditInboxItem] |

### Possible use case
<!-- MANUAL: use_case -->
**Inbox Monitoring**: Check for new messages or mentions to respond to.

**Engagement Tracking**: Monitor comment replies to stay engaged with discussions.
<!-- END MANUAL -->

---

## Get Reddit Post

### What it is
Get detailed information about a specific Reddit post by its ID.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to retrieve complete details about a specific post by its ID. Returns the post title, content, author, score, comment count, and other metadata.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post_id | The ID of the post to fetch (e.g., 'abc123' or full ID 't3_abc123') | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the post couldn't be fetched | str |
| post | Detailed post information | RedditPostDetails |

### Possible use case
<!-- MANUAL: use_case -->
**Post Analysis**: Analyze specific posts for content quality or engagement metrics.

**Content Verification**: Verify post details before interacting with it programmatically.
<!-- END MANUAL -->

---

## Get Reddit Post Comments

### What it is
Get top-level comments on a Reddit post.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to fetch top-level comments on a post. Configure the sort order and limit to control which comments are returned.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post_id | The ID of the post to get comments from | str | Yes |
| limit | Maximum number of top-level comments to fetch (max 100) | int | No |
| sort | Sort order for comments | "best" \| "top" \| "new" \| "controversial" \| "old" \| "qa" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if comments couldn't be fetched | str |
| comment | A comment on the post | RedditComment |
| comments | All fetched comments | List[RedditComment] |
| post_id | The post ID (pass-through for chaining) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Sentiment Analysis**: Analyze comments to gauge community sentiment on a topic.

**Content Moderation**: Review comments for compliance with community guidelines.
<!-- END MANUAL -->

---

## Get Reddit Posts

### What it is
This block fetches Reddit posts from a defined subreddit name.

### How it works
<!-- MANUAL: how_it_works -->
The block connects to Reddit using provided credentials, accesses the specified subreddit, and retrieves posts based on the given parameters. It can limit the number of posts, stop at a specific post, or fetch posts within a certain time frame.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| subreddit | Subreddit name, excluding the /r/ prefix | str | No |
| last_minutes | Post time to stop minutes ago while fetching posts | int | No |
| last_post | Post ID to stop when reached while fetching posts | str | No |
| post_limit | Number of posts to fetch | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post | Reddit post | RedditPost |
| posts | List of all Reddit posts | List[RedditPost] |

### Possible use case
<!-- MANUAL: use_case -->
A content curator could use this block to gather recent posts from a specific subreddit for analysis, summarization, or inclusion in a newsletter.
<!-- END MANUAL -->

---

## Get Reddit User Info

### What it is
Get information about a Reddit user including karma, account age, and verification status.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to retrieve public profile information about a Reddit user, including karma scores, account age, and verification status.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| username | The Reddit username to look up (without /u/ prefix) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if user lookup failed | str |
| user | User information | RedditUserInfo |
| username | The username (pass-through for chaining) | str |

### Possible use case
<!-- MANUAL: use_case -->
**User Verification**: Check user account age and karma before engaging.

**User Research**: Gather user profile data for analysis or outreach decisions.
<!-- END MANUAL -->

---

## Get Subreddit Flairs

### What it is
Get available link flair options for a subreddit.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to retrieve available link flair options for a subreddit. Use the flair IDs with the Create Reddit Post block to apply flair to your posts.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| subreddit | Subreddit name (without /r/ prefix) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if fetch failed | str |
| flair | A flair option | SubredditFlair |
| flairs | All available flairs | List[SubredditFlair] |
| subreddit | The subreddit name (pass-through for chaining) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Post Preparation**: Get available flairs before creating posts to ensure proper categorization.

**Flair Selection**: Present flair options to users or select appropriate flair programmatically.
<!-- END MANUAL -->

---

## Get Subreddit Info

### What it is
Get information about a subreddit including subscriber count, description, and rules.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to retrieve metadata about a subreddit including subscriber count, description, creation date, and posting rules.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| subreddit | Subreddit name (without /r/ prefix) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the subreddit couldn't be fetched | str |
| info | Subreddit information | SubredditInfo |
| subreddit | The subreddit name (pass-through for chaining) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Subreddit Research**: Analyze subreddits before deciding to post or engage.

**Community Analysis**: Compare subreddit sizes and activity for market research.
<!-- END MANUAL -->

---

## Get Subreddit Rules

### What it is
Get the rules for a subreddit to ensure compliance before posting.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to retrieve the posting rules for a subreddit. Review these rules before posting to ensure compliance.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| subreddit | Subreddit name (without /r/ prefix) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if fetch failed | str |
| rule | A subreddit rule | SubredditRule |
| rules | All subreddit rules | List[SubredditRule] |
| subreddit | The subreddit name (pass-through for chaining) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Compliance Check**: Review rules before automated posting to avoid violations.

**Content Guidelines**: Display rules to users before they submit content to a subreddit.
<!-- END MANUAL -->

---

## Get User Posts

### What it is
Fetch posts by a specific Reddit user.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to fetch posts submitted by a specific user. Configure sort order and limit to control which posts are returned.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| username | Reddit username to fetch posts from (without /u/ prefix) | str | Yes |
| post_limit | Maximum number of posts to fetch | int | No |
| sort | Sort order for user posts | "new" \| "hot" \| "top" \| "controversial" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if posts couldn't be fetched | str |
| post | A post by the user | RedditPost |
| posts | All posts by the user | List[RedditPost] |

### Possible use case
<!-- MANUAL: use_case -->
**User Analysis**: Analyze a user's posting history for content patterns or topics.

**Influencer Research**: Research prolific posters in specific communities.
<!-- END MANUAL -->

---

## Instantiate Code Sandbox

### What it is
Instantiate a sandbox environment with internet access in which you can execute code with the Execute Code Step block.

### How it works
<!-- MANUAL: how_it_works -->
This block creates a persistent E2B sandbox environment that can be used for multiple code execution steps. Run setup_commands and setup_code to prepare the environment with dependencies and initial state.

The sandbox persists until its timeout expires or it's explicitly disposed. Use the returned sandbox_id with Execute Code Step blocks for subsequent code execution.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| setup_commands | Shell commands to set up the sandbox before running the code. You can use `curl` or `git` to install your desired Debian based package manager. `pip` and `npm` are pre-installed.  These commands are executed with `sh`, in the foreground. | List[str] | No |
| setup_code | Code to execute in the sandbox | str | No |
| language | Programming language to execute | "python" \| "js" \| "bash" \| "r" \| "java" | No |
| timeout | Execution timeout in seconds | int | No |
| template_id | You can use an E2B sandbox template by entering its ID here. Check out the E2B docs for more details: [E2B - Sandbox template](https://e2b.dev/docs/sandbox-template) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| sandbox_id | ID of the sandbox instance | str |
| response | Text result (if any) of the setup code execution | str |
| stdout_logs | Standard output logs from execution | str |
| stderr_logs | Standard error logs from execution | str |

### Possible use case
<!-- MANUAL: use_case -->
**Complex Pipelines**: Set up an environment with data science libraries for multi-step analysis.

**Persistent State**: Create a sandbox with loaded models or data that multiple workflow branches can access.

**Custom Environments**: Configure specialized environments with specific package versions for reproducible execution.
<!-- END MANUAL -->

---

## Post Reddit Comment

### What it is
This block posts a Reddit comment on a specified Reddit post.

### How it works
<!-- MANUAL: how_it_works -->
The block connects to Reddit using the provided credentials, locates the specified post, and then adds the given comment to that post.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post_id | The ID of the post to comment on | str | Yes |
| comment | The content of the comment to post | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| comment_id | Posted comment ID | str |
| post_id | The post ID (pass-through for chaining) | str |

### Possible use case
<!-- MANUAL: use_case -->
An automated moderation system could use this block to post pre-defined responses or warnings on Reddit posts that violate community guidelines.
<!-- END MANUAL -->

---

## Publish To Medium

### What it is
Publishes a post to Medium.

### How it works
<!-- MANUAL: how_it_works -->
This block publishes articles to Medium using their API. Provide the content in HTML or Markdown format along with a title, tags, and publishing options. The author_id can be obtained from Medium's /me API endpoint.

Configure publish_status to publish immediately, save as draft, or make unlisted. The block returns the published post's ID and URL.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| author_id | The Medium AuthorID of the user. You can get this by calling the /me endpoint of the Medium API.  curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" https://api.medium.com/v1/me  The response will contain the authorId field. | str | No |
| title | The title of your Medium post | str | Yes |
| content | The main content of your Medium post | str | Yes |
| content_format | The format of the content: 'html' or 'markdown' | str | Yes |
| tags | List of tags for your Medium post (up to 5) | List[str] | Yes |
| canonical_url | The original home of this content, if it was originally published elsewhere | str | No |
| publish_status | The publish status | "public" \| "draft" \| "unlisted" | Yes |
| license | The license of the post: 'all-rights-reserved', 'cc-40-by', 'cc-40-by-sa', 'cc-40-by-nd', 'cc-40-by-nc', 'cc-40-by-nc-nd', 'cc-40-by-nc-sa', 'cc-40-zero', 'public-domain' | str | No |
| notify_followers | Whether to notify followers that the user has published | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the post creation failed | str |
| post_id | The ID of the created Medium post | str |
| post_url | The URL of the created Medium post | str |
| published_at | The timestamp when the post was published | int |

### Possible use case
<!-- MANUAL: use_case -->
**Content Syndication**: Automatically publish blog posts or newsletters to Medium to reach a wider audience.

**AI Content Publishing**: Generate articles with AI and publish them directly to Medium.

**Cross-Posting**: Republish existing content from other platforms to Medium with proper canonical URL attribution.
<!-- END MANUAL -->

---

## Read RSS Feed

### What it is
Reads RSS feed entries from a given URL.

### How it works
<!-- MANUAL: how_it_works -->
This block fetches and parses RSS or Atom feeds from a URL. Filter entries by time_period to only get recent items. When run_continuously is enabled, the block polls the feed at the specified polling_rate interval.

Each entry is output individually, enabling processing of new content as it appears. The block also outputs all entries as a list for batch processing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| rss_url | The URL of the RSS feed to read | str | Yes |
| time_period | The time period to check in minutes relative to the run block runtime, e.g. 60 would check for new entries in the last hour. | int | No |
| polling_rate | The number of seconds to wait between polling attempts. | int | Yes |
| run_continuously | Whether to run the block continuously or just once. | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| entry | The RSS item | RSSEntry |
| entries | List of all RSS entries | List[RSSEntry] |

### Possible use case
<!-- MANUAL: use_case -->
**News Monitoring**: Track industry news feeds and process new articles for summarization or alerts.

**Content Aggregation**: Collect posts from multiple RSS feeds for a curated digest or newsletter.

**Blog Triggers**: Monitor a competitor's blog feed to trigger analysis or response workflows.
<!-- END MANUAL -->

---

## Reddit Get My Posts

### What it is
Fetch posts created by the authenticated Reddit user (you).

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to fetch posts you've submitted to Reddit. Useful for managing or analyzing your own posting history.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post_limit | Maximum number of posts to fetch | int | No |
| sort | Sort order for posts | "new" \| "hot" \| "top" \| "controversial" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if posts couldn't be fetched | str |
| post | A post by you | RedditPost |
| posts | All your posts | List[RedditPost] |

### Possible use case
<!-- MANUAL: use_case -->
**Content Management**: Review and manage your Reddit posting history.

**Performance Tracking**: Analyze the engagement of your previous posts.
<!-- END MANUAL -->

---

## Reply To Reddit Comment

### What it is
Reply to a specific Reddit comment. Useful for threaded conversations.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to post a reply to an existing comment. The reply appears as a nested response in the comment thread.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| comment_id | The ID of the comment to reply to | str | Yes |
| reply_text | The text content of the reply | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if reply failed | str |
| comment_id | ID of the newly created reply | str |
| parent_comment_id | The parent comment ID (pass-through for chaining) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Automated Responses**: Reply to comments that mention your product or brand.

**Conversation Engagement**: Participate in discussions by responding to relevant comments.
<!-- END MANUAL -->

---

## Search Reddit

### What it is
Search Reddit for posts matching a query. Can search all of Reddit or a specific subreddit.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to search for posts matching your query. Optionally limit the search to a specific subreddit and configure sort order and time filters.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| query | Search query string | str | Yes |
| subreddit | Limit search to a specific subreddit (without /r/ prefix) | str | No |
| sort | Sort order for search results | "relevance" \| "hot" \| "top" \| "new" \| "comments" | No |
| time_filter | Time filter for search results | "all" \| "day" \| "hour" \| "month" \| "week" \| "year" | No |
| limit | Maximum number of results to return | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if search failed | str |
| result | A search result | RedditSearchResult |
| results | All search results | List[RedditSearchResult] |

### Possible use case
<!-- MANUAL: use_case -->
**Brand Monitoring**: Search for mentions of your product or company across Reddit.

**Topic Research**: Find discussions about specific topics or keywords.
<!-- END MANUAL -->

---

## Send Authenticated Web Request

### What it is
Make an authenticated HTTP request with host-scoped credentials (JSON / form / multipart).

### How it works
<!-- MANUAL: how_it_works -->
This block makes HTTP requests with automatic credential injection based on the request URL's host. Credentials are managed separately and applied when the URL matches a configured host pattern.

Supports JSON, form-encoded, and multipart requests with file uploads. The response is parsed and returned along with separate error outputs for client (4xx) and server (5xx) errors.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL to send the request to | str | Yes |
| method | The HTTP method to use for the request | "GET" \| "POST" \| "PUT" \| "DELETE" \| "PATCH" \| "OPTIONS" \| "HEAD" | No |
| headers | The headers to include in the request | Dict[str, str] | No |
| json_format | If true, send the body as JSON (unless files are also present). | bool | No |
| body | Form/JSON body payload. If files are supplied, this must be a mapping of form‑fields. | Dict[str, Any] | No |
| files_name | The name of the file field in the form data. | str | No |
| files | Mapping of *form field name* → Image url / path / base64 url. | List[str (file)] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Errors for all other exceptions | str |
| response | The response from the server | Response |
| client_error | Errors on 4xx status codes | Client Error |
| server_error | Errors on 5xx status codes | Server Error |

### Possible use case
<!-- MANUAL: use_case -->
**Private API Access**: Call APIs that require authentication without exposing credentials in the workflow.

**OAuth Integrations**: Access protected resources using pre-configured OAuth tokens.

**Multi-Tenant APIs**: Make requests to APIs where credentials vary by host or endpoint.
<!-- END MANUAL -->

---

## Send Email

### What it is
This block sends an email using the provided SMTP credentials.

### How it works
<!-- MANUAL: how_it_works -->
This block sends emails via SMTP using your configured email server credentials. Provide the recipient address, subject, and body content. The SMTP configuration includes server host, port, username, and password.

The block handles connection, authentication, and message delivery, returning a status indicating success or failure.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| to_email | Recipient email address | str | Yes |
| subject | Subject of the email | str | Yes |
| body | Body of the email | str | Yes |
| config | SMTP Config | SMTP Config | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the email sending failed | str |
| status | Status of the email sending operation | str |

### Possible use case
<!-- MANUAL: use_case -->
**Notification Emails**: Send automated notifications when workflow events occur.

**Report Delivery**: Email generated reports or summaries to stakeholders.

**Alert System**: Send email alerts when monitoring workflows detect issues or thresholds.
<!-- END MANUAL -->

---

## Send Reddit Message

### What it is
Send a private message (DM) to a Reddit user.

### How it works
<!-- MANUAL: how_it_works -->
This block uses the Reddit API via PRAW to send a private message to another Reddit user. The message appears in their inbox.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| username | The Reddit username to send a message to (without /u/ prefix) | str | Yes |
| subject | The subject line of the message | str | Yes |
| message | The body content of the message | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if sending failed | str |
| success | Whether the message was sent | bool |
| username | The username (pass-through for chaining) | str |

### Possible use case
<!-- MANUAL: use_case -->
**Outreach**: Send direct messages to users for collaboration or feedback requests.

**Support**: Provide private support or follow-up to users who engaged with your content.
<!-- END MANUAL -->

---

## Send Web Request

### What it is
Make an HTTP request (JSON / form / multipart).

### How it works
<!-- MANUAL: how_it_works -->
This block makes HTTP requests to any URL. Configure the method (GET, POST, PUT, DELETE, PATCH), headers, and request body. Supports JSON, form-encoded, and multipart content types with file uploads.

The response body is parsed and returned. Separate error outputs distinguish between client errors (4xx), server errors (5xx), and other failures.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The URL to send the request to | str | Yes |
| method | The HTTP method to use for the request | "GET" \| "POST" \| "PUT" \| "DELETE" \| "PATCH" \| "OPTIONS" \| "HEAD" | No |
| headers | The headers to include in the request | Dict[str, str] | No |
| json_format | If true, send the body as JSON (unless files are also present). | bool | No |
| body | Form/JSON body payload. If files are supplied, this must be a mapping of form‑fields. | Dict[str, Any] | No |
| files_name | The name of the file field in the form data. | str | No |
| files | Mapping of *form field name* → Image url / path / base64 url. | List[str (file)] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Errors for all other exceptions | str |
| response | The response from the server | Response |
| client_error | Errors on 4xx status codes | Client Error |
| server_error | Errors on 5xx status codes | Server Error |

### Possible use case
<!-- MANUAL: use_case -->
**API Integration**: Call REST APIs to fetch data, trigger actions, or send updates.

**Webhook Delivery**: Send webhook notifications to external services when events occur.

**Custom Services**: Integrate with services that don't have dedicated blocks using their HTTP APIs.
<!-- END MANUAL -->

---

## Transcribe Youtube Video

### What it is
Transcribes a YouTube video using a proxy.

### How it works
<!-- MANUAL: how_it_works -->
This block extracts transcripts from YouTube videos using a proxy service. It parses the YouTube URL to get the video ID and retrieves the available transcript, typically the auto-generated or manually uploaded captions.

The transcript text is returned as a single string, suitable for summarization, analysis, or other text processing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| youtube_url | The URL of the YouTube video to transcribe | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Any error message if the transcription fails | str |
| video_id | The extracted YouTube video ID | str |
| transcript | The transcribed text of the video | str |

### Possible use case
<!-- MANUAL: use_case -->
**Video Summarization**: Extract video transcripts for AI summarization or key point extraction.

**Content Repurposing**: Convert YouTube content into written articles, social posts, or documentation.

**Research Automation**: Transcribe educational or informational videos for analysis and note-taking.
<!-- END MANUAL -->

---
