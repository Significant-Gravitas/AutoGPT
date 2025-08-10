# Publish to Medium

## What it is
The Publish to Medium block is a tool that enables direct publication of content to the Medium platform from within an automated workflow.

## What it does
This block takes a fully formatted blog post, along with associated metadata, and publishes it to Medium using the platform's API. It handles all aspects of the publication process, including setting the title, content, tags, and other post-specific details.

## How it works
The block uses the provided Medium API key and author ID to authenticate with the Medium platform. It then constructs an API request containing all the post details and sends it to Medium's servers. After the post is published, the block retrieves and returns relevant information about the newly created post, such as its unique ID and public URL.

## Inputs
| Input | Description |
|-------|-------------|
| Author ID | The unique identifier for the Medium author account |
| Title | The headline of the Medium post |
| Content | The main body of the post (in HTML or Markdown format) |
| Content Format | Specifies whether the content is in 'html' or 'markdown' format |
| Tags | Up to 5 topic tags to categorize the post (comma-separated) |
| Canonical URL | The original URL if the content was first published elsewhere |
| Publish Status | Sets the post visibility: 'public', 'draft', or 'unlisted' |
| License | The copyright license for the post (default: 'all-rights-reserved') |
| Notify Followers | Boolean flag to notify the author's followers about the new post |
| API Key | The Medium API key for authentication |

## Outputs
| Output | Description |
|--------|-------------|
| Post ID | The unique identifier assigned to the published post by Medium |
| Post URL | The public web address where the post can be viewed |
| Published At | The timestamp indicating when the post was published |
| Error | Any error message returned if the publication process fails |

## Possible use case
A digital marketing team could integrate this block into their content management system to streamline their cross-platform publishing strategy. After creating and approving a blog post in their main system, they could use this block to automatically publish the content to Medium, ensuring consistent and timely distribution across multiple platforms without manual intervention.