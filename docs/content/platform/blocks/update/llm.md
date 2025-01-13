
<file_name>autogpt_platform/backend/backend/blocks/medium.md</file_name>

## Publish to Medium

### What it is
A specialized block that allows users to publish content directly to the Medium platform from within the system.

### What it does
This block enables automated publishing of articles to Medium with customizable settings such as publication status, formatting, and tagging options.

### How it works
The block connects to Medium's API using your credentials, formats your content according to your specifications, and publishes it to your Medium account. It handles all the necessary API communication and provides feedback about the publication process.

### Inputs
- Author ID: Your unique Medium author identifier required for publishing content
- Title: The headline of your Medium post
- Content: The main body of your article
- Content Format: Choose between HTML or Markdown formatting
- Tags: Up to 5 relevant topics or categories for your post
- Canonical URL: Optional link to the original source if the content was previously published elsewhere
- Publish Status: Choose between public, draft, or unlisted visibility
- License: The content license type (e.g., all-rights-reserved, creative commons)
- Notify Followers: Option to alert your followers about the new post
- Credentials: Your Medium API authentication details

### Outputs
- Post ID: The unique identifier for your published Medium post
- Post URL: The direct link to view your published content
- Published At: The timestamp indicating when your post was published
- Error: Any error messages if the publication process fails

### Possible use cases
- Content Marketing Automation: Schedule and publish blog posts automatically to Medium
- Cross-Platform Publishing: Simultaneously publish content across multiple platforms including Medium
- Content Syndication: Republish existing blog content to Medium with proper canonical URL attribution
- Draft Management: Create draft posts for review before public release
- Series Publication: Publish a series of related articles with consistent formatting and tagging

### Notes
- The block supports both HTML and Markdown formatting for flexibility in content creation
- Posts can be published as drafts for review before making them public
- You can control whether your followers are notified of new publications
- The system includes proper error handling and feedback mechanisms
- Various licensing options are available to protect your content rights

