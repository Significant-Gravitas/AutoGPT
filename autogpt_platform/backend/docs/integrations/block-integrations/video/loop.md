# Video Loop
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Loop Video

### What it is
Block to loop a video to a given duration or number of repeats.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| video_in | The input video (can be a URL, data URI, or local path). | str (file) | Yes |
| duration | Target duration (in seconds) to loop the video to. Either duration or n_loops must be provided. | float | No |
| n_loops | Number of times to repeat the video. Either n_loops or duration must be provided. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_out | Looped video returned either as a relative path or a data URI. | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
