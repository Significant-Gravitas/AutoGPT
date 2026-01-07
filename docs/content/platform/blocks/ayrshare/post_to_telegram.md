# Post To Telegram

### What it is
Post to Telegram using Ayrshare.

### What it does
Post to Telegram using Ayrshare

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post | The post text (empty string allowed). Use @handle to mention other Telegram users. | str | No |
| media_urls | Optional list of media URLs. For animated GIFs, only one URL is allowed. Telegram will auto-preview links unless image/video is included. | List[str] | No |
| is_video | Whether the media is a video. Set to true for animated GIFs that don't end in .gif/.GIF extension. | bool | No |
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
