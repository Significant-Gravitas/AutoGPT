# Video Download
<!-- MANUAL: file_description -->
This block downloads videos from URLs, supporting a wide range of video platforms and direct links.
<!-- END MANUAL -->

## Video Download

### What it is
Download video from URL (YouTube, Vimeo, news sites, direct links)

### How it works
<!-- MANUAL: how_it_works -->
The block uses yt-dlp, a powerful video downloading library that supports over 1000 websites. It accepts a URL, quality preference, and output format, then downloads the video while merging the best available video and audio streams for the selected quality. Quality options: **best** (highest available), **1080p/720p/480p** (maximum resolution at that height), **audio_only** (extracts just the audio track).
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
- Downloading source videos for editing or remixing
- Archiving video content for offline processing
- Extracting audio from videos for transcription or podcast creation
- Gathering video content for automated content pipelines
<!-- END MANUAL -->

---
