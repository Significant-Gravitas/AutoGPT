# Ayrshare Post To LinkedIn
<!-- MANUAL: file_description -->
Blocks for posting content to LinkedIn using the Ayrshare social media management API.
<!-- END MANUAL -->

## Post To Linked In

### What it is
Post to LinkedIn using Ayrshare

### How it works
<!-- MANUAL: how_it_works -->
This block uses Ayrshare's social media API to post content to LinkedIn. It handles text posts, images, videos, and documents, with support for scheduling and audience targeting. The block authenticates through Ayrshare's API.

LinkedIn-specific features include visibility controls, comment management, and targeting by country, seniority, industry, and other demographics (requires 300+ followers in target audience).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post | The post text (max 3,000 chars, hashtags supported with #) | str | No |
| media_urls | Optional list of media URLs. LinkedIn supports up to 9 images, videos, or documents (PPT, PPTX, DOC, DOCX, PDF <100MB, <300 pages). | List[str] | No |
| is_video | Whether the media is a video | bool | No |
| schedule_date | UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ) | str (date-time) | No |
| disable_comments | Whether to disable comments | bool | No |
| shorten_links | Whether to shorten links | bool | No |
| unsplash | Unsplash image configuration | str | No |
| requires_approval | Whether to enable approval workflow | bool | No |
| random_post | Whether to generate random post text | bool | No |
| random_media_url | Whether to generate random media | bool | No |
| notes | Additional notes for the post | str | No |
| visibility | Post visibility: 'public' (default), 'connections' (personal only), 'loggedin' | str | No |
| alt_text | Alt text for each image (accessibility feature, not supported for videos/documents) | List[str] | No |
| titles | Title/caption for each image or video | List[str] | No |
| document_title | Title for document posts (max 400 chars, uses filename if not specified) | str | No |
| thumbnail | Thumbnail URL for video (PNG/JPG, same dimensions as video, <10MB) | str | No |
| targeting_countries | Country codes for targeting (e.g., ['US', 'IN', 'DE', 'GB']). Requires 300+ followers in target audience. | List[str] | No |
| targeting_seniorities | Seniority levels for targeting (e.g., ['Senior', 'VP']). Requires 300+ followers in target audience. | List[str] | No |
| targeting_degrees | Education degrees for targeting. Requires 300+ followers in target audience. | List[str] | No |
| targeting_fields_of_study | Fields of study for targeting. Requires 300+ followers in target audience. | List[str] | No |
| targeting_industries | Industry categories for targeting. Requires 300+ followers in target audience. | List[str] | No |
| targeting_job_functions | Job function categories for targeting. Requires 300+ followers in target audience. | List[str] | No |
| targeting_staff_count_ranges | Company size ranges for targeting. Requires 300+ followers in target audience. | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_result | The result of the post | PostResponse |
| post | The result of the post | PostIds |

### Possible use case
<!-- MANUAL: use_case -->
**Thought Leadership**: Automatically share blog posts or industry insights with professional network.

**Scheduled Content**: Queue up a week's worth of LinkedIn posts with scheduled publishing times.

**Targeted Announcements**: Share company updates targeted to specific industries or seniority levels.
<!-- END MANUAL -->

---
