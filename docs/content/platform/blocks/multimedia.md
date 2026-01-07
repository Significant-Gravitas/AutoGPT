# Add Audio To Video

### What it is
Block to attach an audio file to a video file using moviepy.

### What it does
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
| output_return_type | Return the final output as a relative path or base64 data URI. | "file_path" | "data_uri" | No |

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

## Loop Video

### What it is
Block to loop a video to a given duration or number of repeats.

### What it does
Block to loop a video to a given duration or number of repeats.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| video_in | The input video (can be a URL, data URI, or local path). | str (file) | Yes |
| duration | Target duration (in seconds) to loop the video to. If omitted, defaults to no looping. | float | No |
| n_loops | Number of times to repeat the video. If omitted, defaults to 1 (no repeat). | int | No |
| output_return_type | How to return the output video. Either a relative path or base64 data URI. | "file_path" | "data_uri" | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_out | Looped video returned either as a relative path or a data URI. | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Media Duration

### What it is
Block to get the duration of a media file.

### What it does
Block to get the duration of a media file.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
