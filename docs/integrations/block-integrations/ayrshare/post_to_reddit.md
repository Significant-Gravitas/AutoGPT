# Ayrshare Post To Reddit
<!-- MANUAL: file_description -->
Blocks for posting content to Reddit using the Ayrshare social media management API.
<!-- END MANUAL -->

## Post To Reddit

### What it is
Post to Reddit using Ayrshare

### How it works
<!-- MANUAL: how_it_works -->
This block uses Ayrshare's API to publish content to Reddit. It supports text posts, image posts, and video submissions with optional scheduling and link shortening features.

The block authenticates through Ayrshare and submits content to your connected Reddit account. Common options include approval workflows for content review before publishing, random content generation, and Unsplash integration for sourcing images.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post | The post text to be published | str | No |
| media_urls | Optional list of media URLs to include. Set is_video in advanced settings to true if you want to upload videos. | List[str] | No |
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
**Community Engagement**: Share relevant content to niche subreddits as part of community marketing strategies.

**Content Distribution**: Cross-post blog articles or announcements to relevant Reddit communities for broader reach.

**Brand Monitoring Response**: Automatically share updates or responses in communities where your brand is discussed.
<!-- END MANUAL -->

---
