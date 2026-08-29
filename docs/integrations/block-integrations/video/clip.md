# Video Clip
<!-- MANUAL: file_description -->
This block extracts a specific time segment from a video file, allowing you to trim videos to precise start and end times.
<!-- END MANUAL -->

## Video Clip

### What it is
Extract a time segment from a video

### How it works
<!-- MANUAL: how_it_works -->
The block uses MoviePy's `subclipped` function to extract a portion of the video between specified start and end times. It validates that end time is greater than start time, then creates a new video file containing only the selected segment. The output is encoded with H.264 video codec and AAC audio codec, preserving both video and audio from the original clip.
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
- Extracting highlights from a longer video
- Trimming intro/outro from recorded content
- Creating short clips for social media from longer videos
- Isolating specific segments for further processing in a workflow
<!-- END MANUAL -->

---
