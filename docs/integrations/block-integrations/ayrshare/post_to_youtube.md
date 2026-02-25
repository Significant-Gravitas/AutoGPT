# Ayrshare Post To YouTube
<!-- MANUAL: file_description -->
Blocks for uploading videos to YouTube using the Ayrshare social media management API.
<!-- END MANUAL -->

## Post To You Tube

### What it is
Post to YouTube using Ayrshare

### How it works
<!-- MANUAL: how_it_works -->
This block uses Ayrshare's API to upload videos to YouTube. It handles video uploads with extensive metadata including titles, descriptions, tags, custom thumbnails, playlist assignment, category selection, and visibility controls (public, private, unlisted).

The block supports YouTube Shorts (up to 3 minutes), geographic targeting to allow or block specific countries, subtitle files (SRT/SBV format), synthetic/AI content disclosure, kids content labeling, and subscriber notification controls. Videos can be scheduled for specific publish times.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post | Video description (max 5,000 chars, empty string allowed). Cannot contain < or > characters. | str | Yes |
| media_urls | Required video URL. YouTube only supports 1 video per post. | List[str] | No |
| is_video | Whether the media is a video | bool | No |
| schedule_date | UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ) | str (date-time) | No |
| disable_comments | Whether to disable comments | bool | No |
| shorten_links | Whether to shorten links | bool | No |
| unsplash | Unsplash image configuration | str | No |
| requires_approval | Whether to enable approval workflow | bool | No |
| random_post | Whether to generate random post text | bool | No |
| random_media_url | Whether to generate random media | bool | No |
| notes | Additional notes for the post | str | No |
| title | Video title (max 100 chars, required). Cannot contain < or > characters. | str | Yes |
| visibility | Video visibility: 'private' (default), 'public' , or 'unlisted' | "private" \| "public" \| "unlisted" | No |
| thumbnail | Thumbnail URL (JPEG/PNG under 2MB, must end in .png/.jpg/.jpeg). Requires phone verification. | str | No |
| playlist_id | Playlist ID to add video (user must own playlist) | str | No |
| tags | Video tags (min 2 chars each, max 500 chars total) | List[str] | No |
| made_for_kids | Self-declared kids content | bool | No |
| is_shorts | Post as YouTube Short (max 3 minutes, adds #shorts) | bool | No |
| notify_subscribers | Send notification to subscribers | bool | No |
| category_id | Video category ID (e.g., 24 = Entertainment) | int | No |
| contains_synthetic_media | Disclose realistic AI/synthetic content | bool | No |
| publish_at | UTC publish time (YouTube controlled, format: 2022-10-08T21:18:36Z) | str | No |
| targeting_block_countries | Country codes to block from viewing (e.g., ['US', 'CA']) | List[str] | No |
| targeting_allow_countries | Country codes to allow viewing (e.g., ['GB', 'AU']) | List[str] | No |
| subtitle_url | URL to SRT or SBV subtitle file (must be HTTPS and end in .srt/.sbv, under 100MB) | str | No |
| subtitle_language | Language code for subtitles (default: 'en') | str | No |
| subtitle_name | Name of caption track (max 150 chars, default: 'English') | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_result | The result of the post | PostResponse |
| post | The result of the post | PostIds |

### Possible use case
<!-- MANUAL: use_case -->
**Video Publishing Pipeline**: Automate video uploads with thumbnails, descriptions, and playlist organization for content creators.

**YouTube Shorts Automation**: Publish short-form vertical videos to YouTube Shorts with proper metadata and hashtags.

**Multi-Region Content**: Upload videos with geographic restrictions for region-specific content licensing or compliance.
<!-- END MANUAL -->

---
