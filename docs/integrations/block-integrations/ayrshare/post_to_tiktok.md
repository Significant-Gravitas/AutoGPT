# Ayrshare Post To TikTok
<!-- MANUAL: file_description -->
Blocks for posting videos and image slideshows to TikTok using the Ayrshare social media management API.
<!-- END MANUAL -->

## Post To Tik Tok

### What it is
Post to TikTok using Ayrshare

### How it works
<!-- MANUAL: how_it_works -->
This block uses Ayrshare's API to publish content to TikTok. It supports video posts and image slideshows (up to 35 images), with extensive options for content labeling including AI-generated disclosure, branded content, and brand organic content tags.

The block connects to TikTok's API through Ayrshare with controls for visibility, duet/stitch permissions, comment settings, auto-music, and thumbnail selection. Videos can be posted as drafts for final review, and scheduled for future publishing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post | The post text (max 2,200 chars, empty string allowed). Use @handle to mention users. Line breaks will be ignored. | str | Yes |
| media_urls | Required media URLs. Either 1 video OR up to 35 images (JPG/JPEG/WEBP only). Cannot mix video and images. | List[str] | No |
| is_video | Whether the media is a video | bool | No |
| schedule_date | UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ) | str (date-time) | No |
| disable_comments | Disable comments on the published post | bool | No |
| shorten_links | Whether to shorten links | bool | No |
| unsplash | Unsplash image configuration | str | No |
| requires_approval | Whether to enable approval workflow | bool | No |
| random_post | Whether to generate random post text | bool | No |
| random_media_url | Whether to generate random media | bool | No |
| notes | Additional notes for the post | str | No |
| auto_add_music | Whether to automatically add recommended music to the post. If you set this field to true, you can change the music later in the TikTok app. | bool | No |
| disable_duet | Disable duets on published video (video only) | bool | No |
| disable_stitch | Disable stitch on published video (video only) | bool | No |
| is_ai_generated | If you enable the toggle, your video will be labeled as “Creator labeled as AI-generated” once posted and can’t be changed. The “Creator labeled as AI-generated” label indicates that the content was completely AI-generated or significantly edited with AI. | bool | No |
| is_branded_content | Whether to enable the Branded Content toggle. If this field is set to true, the video will be labeled as Branded Content, indicating you are in a paid partnership with a brand. A “Paid partnership” label will be attached to the video. | bool | No |
| is_brand_organic | Whether to enable the Brand Organic Content toggle. If this field is set to true, the video will be labeled as Brand Organic Content, indicating you are promoting yourself or your own business. A “Promotional content” label will be attached to the video. | bool | No |
| image_cover_index | Index of image to use as cover (0-based, image posts only) | int | No |
| title | Title for image posts | str | No |
| thumbnail_offset | Video thumbnail frame offset in milliseconds (video only) | int | No |
| visibility | Post visibility: 'public', 'private', 'followers', or 'friends' | "public" \| "private" \| "followers" | No |
| draft | Create as draft post (video only) | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_result | The result of the post | PostResponse |
| post | The result of the post | PostIds |

### Possible use case
<!-- MANUAL: use_case -->
**Creator Content Pipeline**: Automate video uploads with proper AI disclosure labels and visibility settings for content creators.

**Brand Campaigns**: Publish branded content with proper disclosure labels to maintain FTC compliance and platform guidelines.

**Image Slideshow Posts**: Create TikTok slideshows from product images or photo series with automatic cover selection.
<!-- END MANUAL -->

---
