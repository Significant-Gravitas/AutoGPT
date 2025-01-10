
## Publish to Medium

### What it is
A block that enables automated publishing of content to the Medium platform, allowing users to create and manage posts programmatically.

### What it does
This block takes your content and various publishing preferences and automatically posts it to Medium under your account. It can create posts in different states (public, draft, or unlisted) and supports various formatting and customization options.

### How it works
The block connects to your Medium account using your API credentials, takes your content and publishing preferences, and creates a new post on Medium according to your specifications. It handles all the necessary communication with Medium's servers and returns information about the published post.

### Inputs
- Author ID: Your unique Medium author identifier, required to specify where the post should be published
- Title: The headline of your Medium post
- Content: The main body of your post (can be in HTML or Markdown format)
- Content Format: Specifies whether your content is in HTML or Markdown format
- Tags: Up to 5 topic tags to help categorize your post
- Canonical URL: Optional link to the original source if this content was published elsewhere first
- Publish Status: Choose between public (visible to all), draft (private), or unlisted (visible only with direct link)
- License: The type of content license for your post (e.g., all rights reserved, Creative Commons)
- Notify Followers: Option to notify your followers when you publish the post
- Credentials: Your Medium API authentication details

### Outputs
- Post ID: The unique identifier assigned to your published post
- Post URL: The direct web link to your published post
- Published At: The timestamp indicating when the post was published
- Error: Any error message if the post creation fails

### Possible use case
A content marketing team could use this block to automatically publish their blog posts to Medium as part of their content distribution workflow. For example, when a new blog post is approved on their main website, this block could automatically create a copy on Medium with proper attribution (using the canonical URL), helping to reach a wider audience while maintaining SEO benefits.

Additional features like setting posts as drafts first for review, adding relevant tags for better discoverability, and controlling follower notifications make it a versatile tool for content management and distribution strategies.

