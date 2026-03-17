# Ayrshare Post To Instagram
<!-- MANUAL: file_description -->
Blocks for posting content to Instagram using the Ayrshare social media management API.
<!-- END MANUAL -->

## Post To Instagram

### What it is
Post to Instagram using Ayrshare. Requires a Business or Creator Instagram Account connected with a Facebook Page

### How it works
<!-- MANUAL: how_it_works -->
This block uses Ayrshare's API to publish content to Instagram Business or Creator accounts. It supports feed posts, Stories (24-hour expiration), Reels, and carousels (up to 10 images/videos), with features like collaborator invitations, location tagging, and user tags with coordinates.

The block requires an Instagram account connected to a Facebook Page and authenticates through Meta's Graph API via Ayrshare. Instagram-specific features include auto-resize for optimal dimensions, audio naming for Reels, and thumbnail customization with frame offset control.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post | The post text (max 2,200 chars, up to 30 hashtags, 3 @mentions) | str | No |
| media_urls | Optional list of media URLs. Instagram supports up to 10 images/videos in a carousel. | List[str] | No |
| is_video | Whether the media is a video | bool | No |
| schedule_date | UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ) | str (date-time) | No |
| disable_comments | Whether to disable comments | bool | No |
| shorten_links | Whether to shorten links | bool | No |
| unsplash | Unsplash image configuration | str | No |
| requires_approval | Whether to enable approval workflow | bool | No |
| random_post | Whether to generate random post text | bool | No |
| random_media_url | Whether to generate random media | bool | No |
| notes | Additional notes for the post | str | No |
| is_story | Whether to post as Instagram Story (24-hour expiration) | bool | No |
| share_reels_feed | Whether Reel should appear in both Feed and Reels tabs | bool | No |
| audio_name | Audio name for Reels (e.g., 'The Weeknd - Blinding Lights') | str | No |
| thumbnail | Thumbnail URL for Reel video | str | No |
| thumbnail_offset | Thumbnail frame offset in milliseconds (default: 0) | int | No |
| alt_text | Alt text for each media item (up to 1,000 chars each, accessibility feature), each item in the list corresponds to a media item in the media_urls list | List[str] | No |
| location_id | Facebook Page ID or name for location tagging (e.g., '7640348500' or '@guggenheimmuseum') | str | No |
| user_tags | List of users to tag with coordinates for images | List[Dict[str, Any]] | No |
| collaborators | Instagram usernames to invite as collaborators (max 3, public accounts only) | List[str] | No |
| auto_resize | Auto-resize images to 1080x1080px for Instagram | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_result | The result of the post | PostResponse |
| post | The result of the post | PostIds |

### Possible use case
<!-- MANUAL: use_case -->
**Influencer Collaborations**: Create posts with collaborator tags to feature brand partnerships across multiple accounts.

**E-commerce Product Showcases**: Share carousel posts of product images with location tags for local discovery.

**Reels Automation**: Automatically publish short-form video content with custom thumbnails and trending audio.
<!-- END MANUAL -->

---
