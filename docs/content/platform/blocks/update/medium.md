
## Publish to Medium Block

### What it is
A functional block that enables automated publishing of content to the Medium platform, allowing users to create and manage posts programmatically.

### What it does
This block publishes content to Medium with customizable settings, including the ability to set the post's status (public, draft, or unlisted), add tags, and configure various publishing options.

### How it works
The block connects to the Medium API using provided credentials, processes the user's content and settings, and creates a new post on the platform. It handles the entire publishing workflow, from authentication to post creation, and returns information about the published content.

### Inputs
- Author ID: The unique identifier for the Medium author account where the content will be published
- Title: The headline or title of the Medium post
- Content: The main body of the post, which can be in HTML or Markdown format
- Content Format: Specifies whether the content is formatted in HTML or Markdown
- Tags: Up to 5 topic tags that categorize the post (helps with content discovery)
- Canonical URL: Optional link to the original source if the content was published elsewhere first
- Publish Status: Sets whether the post should be public, draft, or unlisted
- License: The content license type (e.g., all-rights-reserved, creative commons options)
- Notify Followers: Option to notify followers about the new post
- Credentials: Medium API authentication details needed to publish content

### Outputs
- Post ID: The unique identifier assigned to the published post
- Post URL: The direct web link to access the published content on Medium
- Published At: The timestamp indicating when the post was published
- Error: Any error message if the publishing process fails

### Possible use cases
- Content syndication: Automatically republish blog posts from your website to Medium
- Content distribution: Share company updates or announcements across platforms
- Draft management: Create draft posts for review before public release
- Cross-platform publishing: Part of a workflow that publishes content to multiple platforms
- Automated content scheduling: Schedule posts to be published at specific times
- Content archiving: Maintain a backup of your content on Medium

### Notes
- The block supports different publishing states (public, draft, unlisted) for flexible content management
- You can control whether followers are notified about new posts
- The block handles proper attribution through canonical URLs
- Multiple license options are available to protect your content rights
- Error handling is built-in to provide clear feedback if publishing fails

