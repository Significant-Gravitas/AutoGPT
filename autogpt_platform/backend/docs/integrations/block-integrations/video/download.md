# Video Download
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Video Download

### What it is
Download video from URL (YouTube, Vimeo, news sites, direct links)

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | URL of the video to download (YouTube, Vimeo, direct link, etc.) | str | Yes |
| quality | Video quality preference | "best" \| "1080p" \| "720p" \| "480p" \| "audio_only" | No |
| output_format | Output video format | "mp4" \| "webm" \| "mkv" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_file | Downloaded video (path or data URI) | str (file) |
| duration | Video duration in seconds | float |
| title | Video title from source | str |
| source_url | Original source URL | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
