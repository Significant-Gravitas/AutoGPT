# Video Duration
<!-- MANUAL: file_description -->
This block retrieves the duration of video or audio files, useful for planning and conditional logic in media workflows.
<!-- END MANUAL -->

## Media Duration

### What it is
Block to get the duration of a media file.

### How it works
<!-- MANUAL: how_it_works -->
The block uses MoviePy to load the media file and extract its duration property. It supports both video files (using VideoFileClip) and audio files (using AudioFileClip), determined by the `is_video` flag. The media can be provided as a URL, data URI, or local file path. The duration is returned in seconds as a floating-point number.
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
- Checking video length before processing to avoid timeout issues
- Calculating how many times to loop a video to reach a target duration
- Validating that uploaded content meets length requirements
- Building conditional workflows based on media duration
<!-- END MANUAL -->

---
