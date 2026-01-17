# Ayrshare Post To Snapchat
<!-- MANUAL: file_description -->
Blocks for posting video content to Snapchat using the Ayrshare social media management API.
<!-- END MANUAL -->

## Post To Snapchat

### What it is
Post to Snapchat using Ayrshare

### How it works
<!-- MANUAL: how_it_works -->
This block uses Ayrshare's API to publish video content to Snapchat. Snapchat only supports video content, with three destination options: Stories (24-hour ephemeral content), Saved Stories (persistent Stories), and Spotlight (public discovery feed).

The block authenticates through Ayrshare and uploads video content with optional custom thumbnails. Videos can be scheduled for future publishing and support approval workflows for content review before going live.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post | The post text (optional for video-only content) | str | No |
| media_urls | Required video URL for Snapchat posts. Snapchat only supports video content. | List[str] | No |
| is_video | Whether the media is a video | bool | No |
| schedule_date | UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ) | str (date-time) | No |
| disable_comments | Whether to disable comments | bool | No |
| shorten_links | Whether to shorten links | bool | No |
| unsplash | Unsplash image configuration | str | No |
| requires_approval | Whether to enable approval workflow | bool | No |
| random_post | Whether to generate random post text | bool | No |
| random_media_url | Whether to generate random media | bool | No |
| notes | Additional notes for the post | str | No |
| story_type | Type of Snapchat content: 'story' (24-hour Stories), 'saved_story' (Saved Stories), or 'spotlight' (Spotlight posts) | str | No |
| video_thumbnail | Thumbnail URL for video content (optional, auto-generated if not provided) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_result | The result of the post | PostResponse |
| post | The result of the post | PostIds |

### Possible use case
<!-- MANUAL: use_case -->
**Ephemeral Marketing**: Share time-sensitive promotions or behind-the-scenes content that creates urgency through 24-hour Stories.

**Public Discovery**: Post engaging video content to Spotlight to reach new audiences beyond your followers.

**Scheduled Story Series**: Plan and schedule a sequence of video Stories for product launches or events.
<!-- END MANUAL -->

---
