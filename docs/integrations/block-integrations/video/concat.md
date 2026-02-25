# Video Concat
<!-- MANUAL: file_description -->
This block merges multiple video clips into a single continuous video, with optional transitions between clips.
<!-- END MANUAL -->

## Video Concat

### What it is
Merge multiple video clips into one continuous video

### How it works
<!-- MANUAL: how_it_works -->
The block uses MoviePy's `concatenate_videoclips` function to join multiple videos in sequence. It supports three transition modes: **none** (direct concatenation), **crossfade** (smooth blending where clips overlap), and **fade_black** (each clip fades out to black and the next fades in). At least 2 videos are required. The output is encoded with H.264 video codec and AAC audio codec.
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
- Combining multiple clips into a compilation video
- Assembling intro, main content, and outro segments
- Creating montages from multiple source videos
- Building video playlists or slideshows with transitions
<!-- END MANUAL -->

---
