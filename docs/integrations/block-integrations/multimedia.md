# Multimedia
<!-- MANUAL: file_description -->
Blocks for processing and manipulating video and audio files.
<!-- END MANUAL -->

## Add Audio To Video

### What it is
Block to attach an audio file to a video file using moviepy.

### How it works
<!-- MANUAL: how_it_works -->
This block combines a video file with an audio file using the moviepy library. The audio track is attached to the video, optionally with volume adjustment via the volume parameter (1.0 = original volume).

Input files can be URLs, data URIs, or local paths. The output format is automatically determined: `workspace://` URLs in CoPilot, data URIs in graph executions.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| video_in | Video input (URL, data URI, or local path). | str (file) | Yes |
| audio_in | Audio input (URL, data URI, or local path). | str (file) | Yes |
| volume | Volume scale for the newly attached audio track (1.0 = original). | float | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_out | Final video (with attached audio), as a path or data URI. | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
**Add Voiceover**: Combine generated voiceover audio with video content for narrated videos.

**Background Music**: Add music tracks to silent videos or replace existing audio.

**Audio Replacement**: Swap the audio track of a video for localization or accessibility.
<!-- END MANUAL -->

---

## Loop Video

### What it is
Block to loop a video to a given duration or number of repeats.

### How it works
<!-- MANUAL: how_it_works -->
This block extends a video by repeating it to reach a target duration or number of loops. Set duration to specify the total length in seconds, or use n_loops to repeat the video a specific number of times.

The looped video is seamlessly concatenated. The output format is automatically determined: `workspace://` URLs in CoPilot, data URIs in graph executions.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| video_in | The input video (can be a URL, data URI, or local path). | str (file) | Yes |
| duration | Target duration (in seconds) to loop the video to. If omitted, defaults to no looping. | float | No |
| n_loops | Number of times to repeat the video. If omitted, defaults to 1 (no repeat). | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_out | Looped video returned either as a relative path or a data URI. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Background Videos**: Loop short clips to match the duration of longer audio or content.

**GIF-Like Content**: Create seamlessly looping video content for social media.

**Filler Content**: Extend short video clips to meet minimum duration requirements.
<!-- END MANUAL -->

---

## Media Duration

### What it is
Block to get the duration of a media file.

### How it works
<!-- MANUAL: how_it_works -->
This block analyzes a media file and returns its duration in seconds. Set is_video to true for video files or false for audio files to ensure proper parsing.

The input can be a URL, data URI, or local file path. The duration is returned as a float for precise timing calculations.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| media_in | Media input (URL, data URI, or local path). | str (file) | Yes |
| is_video | Whether the media is a video (True) or audio (False). | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| duration | Duration of the media file (in seconds). | float |

### Possible use case
<!-- MANUAL: use_case -->
**Video Processing Prep**: Get video duration before deciding how to loop, trim, or synchronize it.

**Audio Matching**: Determine audio length to generate matching-length video content.

**Content Validation**: Verify that uploaded media meets duration requirements.
<!-- END MANUAL -->

---
