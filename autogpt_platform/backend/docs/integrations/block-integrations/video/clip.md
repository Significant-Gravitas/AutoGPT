# Video Clip
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Video Clip

### What it is
Extract a time segment from a video

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| video_in | Input video (URL, data URI, or local path) | str (file) | Yes |
| start_time | Start time in seconds | float | Yes |
| end_time | End time in seconds | float | Yes |
| output_format | Output format | "mp4" \| "webm" \| "mkv" \| "mov" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_out | Clipped video file (path or data URI) | str (file) |
| duration | Clip duration in seconds | float |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
