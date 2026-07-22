# Video Concat
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Video Concat

### What it is
Merge multiple video clips into one continuous video

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| videos | List of video files to concatenate (in order) | List[str (file)] | Yes |
| transition | Transition between clips | "none" \| "crossfade" \| "fade_black" | No |
| transition_duration | Transition duration in seconds | int | No |
| output_format | Output format | "mp4" \| "webm" \| "mkv" \| "mov" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_out | Concatenated video file (path or data URI) | str (file) |
| total_duration | Total duration in seconds | float |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
