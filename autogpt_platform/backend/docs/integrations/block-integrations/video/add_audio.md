# Video Add Audio
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Add Audio To Video

### What it is
Block to attach an audio file to a video file using moviepy.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
