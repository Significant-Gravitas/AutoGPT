# Ayrshare Post To Threads
<!-- MANUAL: file_description -->
Blocks for posting content to Threads using the Ayrshare social media management API.
<!-- END MANUAL -->

## Post To Threads

### What it is
Post to Threads using Ayrshare

### How it works
<!-- MANUAL: how_it_works -->
This block uses Ayrshare's API to publish content to Threads (Meta's text-based social platform). It supports text posts (up to 500 characters with one hashtag), images, videos, and carousels (up to 20 items), with automatic link previews when no media is attached.

The block authenticates through Meta's API via Ayrshare. Content can mention users via @handle syntax, be scheduled for future publishing, and include approval workflows for content review.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post | The post text (max 500 chars, empty string allowed). Only 1 hashtag allowed. Use @handle to mention users. | str | No |
| media_urls | Optional list of media URLs. Supports up to 20 images/videos in a carousel. Auto-preview links unless media is included. | List[str] | No |
| is_video | Whether the media is a video | bool | No |
| schedule_date | UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ) | str (date-time) | No |
| disable_comments | Whether to disable comments | bool | No |
| shorten_links | Whether to shorten links | bool | No |
| unsplash | Unsplash image configuration | str | No |
| requires_approval | Whether to enable approval workflow | bool | No |
| random_post | Whether to generate random post text | bool | No |
| random_media_url | Whether to generate random media | bool | No |
| notes | Additional notes for the post | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_result | The result of the post | PostResponse |
| post | The result of the post | PostIds |

### Possible use case
<!-- MANUAL: use_case -->
**Thought Leadership**: Share quick insights, opinions, or industry commentary in a conversational format.

**Cross-Platform Text Content**: Automatically syndicate text-based content from other platforms to Threads.

**Community Engagement**: Post discussion prompts or responses to engage with your Threads audience.
<!-- END MANUAL -->

---
