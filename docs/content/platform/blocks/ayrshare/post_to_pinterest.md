# Post To Pinterest

### What it is
Post to Pinterest using Ayrshare.

### What it does
Post to Pinterest using Ayrshare

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post | Pin description (max 500 chars, links not clickable - use link field instead) | str | No |
| media_urls | Required image/video URLs. Pinterest requires at least one image. Videos need thumbnail. Up to 5 images for carousel. | List[str] | No |
| is_video | Whether the media is a video | bool | No |
| schedule_date | UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ) | str (date-time) | No |
| disable_comments | Whether to disable comments | bool | No |
| shorten_links | Whether to shorten links | bool | No |
| unsplash | Unsplash image configuration | str | No |
| requires_approval | Whether to enable approval workflow | bool | No |
| random_post | Whether to generate random post text | bool | No |
| random_media_url | Whether to generate random media | bool | No |
| notes | Additional notes for the post | str | No |
| pin_title | Pin title displayed in 'Add your title' section (max 100 chars) | str | No |
| link | Clickable destination URL when users click the pin (max 2048 chars) | str | No |
| board_id | Pinterest Board ID to post to (from /user/details endpoint, uses default board if not specified) | str | No |
| note | Private note for the pin (only visible to you and board collaborators) | str | No |
| thumbnail | Required thumbnail URL for video pins (must have valid image Content-Type) | str | No |
| carousel_options | Options for each image in carousel (title, link, description per image) | List[PinterestCarouselOption] | No |
| alt_text | Alt text for each image/video (max 500 chars each, accessibility feature) | List[str] | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_result | The result of the post | PostResponse |
| post | The result of the post | PostIds |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
