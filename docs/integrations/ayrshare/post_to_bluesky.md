# Ayrshare Post To Bluesky
<!-- MANUAL: file_description -->
Blocks for posting content to Bluesky using the Ayrshare social media management API.
<!-- END MANUAL -->

## Post To Bluesky

### What it is
Post to Bluesky using Ayrshare

### How it works
<!-- MANUAL: how_it_works -->
This block uses Ayrshare's social media API to publish content to Bluesky. It handles text posts (up to 300 characters), images (up to 4), and video content with support for scheduling, accessibility features like alt text, and link shortening.

The block authenticates through your Ayrshare credentials and sends the post data to Ayrshare's unified API, which then publishes to Bluesky. It returns post identifiers and status information upon completion, or error details if the operation fails.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post | The post text to be published (max 300 characters for Bluesky) | str | No |
| media_urls | Optional list of media URLs to include. Bluesky supports up to 4 images or 1 video. | List[str] | No |
| is_video | Whether the media is a video | bool | No |
| schedule_date | UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ) | str (date-time) | No |
| disable_comments | Whether to disable comments | bool | No |
| shorten_links | Whether to shorten links | bool | No |
| unsplash | Unsplash image configuration | str | No |
| requires_approval | Whether to enable approval workflow | bool | No |
| random_post | Whether to generate random post text | bool | No |
| random_media_url | Whether to generate random media | bool | No |
| notes | Additional notes for the post | str | No |
| alt_text | Alt text for each media item (accessibility) | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_result | The result of the post | PostResponse |
| post | The result of the post | PostIds |

### Possible use case
<!-- MANUAL: use_case -->
**Cross-Platform Publishing**: Automatically share content across Bluesky and other social networks from a single workflow.

**Scheduled Content Calendar**: Queue up posts with specific publishing times to maintain consistent presence on Bluesky.

**Visual Content Sharing**: Share image galleries with accessibility-friendly alt text for photo-focused content strategies.
<!-- END MANUAL -->

---
