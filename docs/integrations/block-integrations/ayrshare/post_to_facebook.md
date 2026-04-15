# Ayrshare Post To Facebook
<!-- MANUAL: file_description -->
Blocks for posting content to Facebook Pages using the Ayrshare social media management API.
<!-- END MANUAL -->

## Post To Facebook

### What it is
Post to Facebook using Ayrshare

### How it works
<!-- MANUAL: how_it_works -->
This block uses Ayrshare's social media API to publish content to Facebook Pages. It supports text posts, images, videos, carousels (2-10 items), Reels, and Stories, with features like audience targeting by age and country, location tagging, and scheduling.

The block authenticates through Ayrshare and leverages the Meta Graph API to handle various Facebook-specific formats. Advanced options include draft mode for Meta Business Suite, custom link previews, and video thumbnails. Results include post IDs for tracking engagement.
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
| is_carousel | Whether to post a carousel | bool | No |
| carousel_link | The URL for the 'See More At' button in the carousel | str | No |
| carousel_items | List of carousel items with name, link and picture URLs. Min 2, max 10 items. | List[CarouselItem] | No |
| is_reels | Whether to post to Facebook Reels | bool | No |
| reels_title | Title for the Reels video (max 255 chars) | str | No |
| reels_thumbnail | Thumbnail URL for Reels video (JPEG/PNG, <10MB) | str | No |
| is_story | Whether to post as a Facebook Story | bool | No |
| media_captions | Captions for each media item | List[str] | No |
| location_id | Facebook Page ID or name for location tagging | str | No |
| age_min | Minimum age for audience targeting (13,15,18,21,25) | int | No |
| target_countries | List of country codes to target (max 25) | List[str] | No |
| alt_text | Alt text for each media item | List[str] | No |
| video_title | Title for video post | str | No |
| video_thumbnail | Thumbnail URL for video post | str | No |
| is_draft | Save as draft in Meta Business Suite | bool | No |
| scheduled_publish_date | Schedule publish time in Meta Business Suite (UTC) | str | No |
| preview_link | URL for custom link preview | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_result | The result of the post | PostResponse |
| post | The result of the post | PostIds |

### Possible use case
<!-- MANUAL: use_case -->
**Product Launches**: Create carousel posts showcasing multiple product images with links to purchase pages.

**Event Promotion**: Share event details with age-targeted reach and location tagging for local business events.

**Short-Form Video**: Automatically publish Reels with custom thumbnails to maximize video content reach.
<!-- END MANUAL -->

---
