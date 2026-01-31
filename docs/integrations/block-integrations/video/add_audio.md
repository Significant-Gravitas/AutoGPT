# Video Add Audio
<!-- MANUAL: file_description -->
This block allows you to attach a separate audio track to a video file, replacing or combining with the original audio.
<!-- END MANUAL -->

## Add Audio To Video

### What it is
Block to attach an audio file to a video file using moviepy.

### How it works
<!-- MANUAL: how_it_works -->
The block uses MoviePy to combine video and audio files. It loads the video and audio inputs (which can be URLs, data URIs, or local paths), optionally scales the audio volume, then writes the combined result to a new video file using H.264 video codec and AAC audio codec.
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
- Adding background music to a silent screen recording
- Replacing original audio with a voiceover or translated audio track
- Combining AI-generated speech with stock footage
- Adding sound effects to video content
<!-- END MANUAL -->

---
