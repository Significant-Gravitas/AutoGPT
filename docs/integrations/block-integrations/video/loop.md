# Video Loop
<!-- MANUAL: file_description -->
This block repeats a video to extend its duration, either to a specific length or a set number of repetitions.
<!-- END MANUAL -->

## Loop Video

### What it is
Block to loop a video to a given duration or number of repeats.

### How it works
<!-- MANUAL: how_it_works -->
The block uses MoviePy's Loop effect to repeat a video clip. You can specify either a target duration (the video will repeat until reaching that length) or a number of loops (the video will repeat that many times). The Loop effect handles both video and audio looping automatically, maintaining sync. Either `duration` or `n_loops` must be provided. The output is encoded with H.264 video codec and AAC audio codec.
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
- Extending a short background video to match the length of narration audio
- Creating seamless looping content for digital signage
- Repeating a product demo video multiple times for emphasis
- Extending short clips to meet minimum duration requirements for platforms
<!-- END MANUAL -->

---
