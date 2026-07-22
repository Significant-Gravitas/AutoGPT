# Video Narration
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Video Narration

### What it is
Generate AI narration and add to video

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| video_in | Input video (URL, data URI, or local path) | str (file) | Yes |
| script | Narration script text | str | Yes |
| voice_id | ElevenLabs voice ID | str | No |
| model_id | ElevenLabs TTS model | "eleven_multilingual_v2" \| "eleven_flash_v2_5" \| "eleven_turbo_v2_5" \| "eleven_turbo_v2" | No |
| mix_mode | How to combine with original audio. 'ducking' applies stronger attenuation than 'mix'. | "replace" \| "mix" \| "ducking" | No |
| narration_volume | Narration volume (0.0 to 2.0) | float | No |
| original_volume | Original audio volume when mixing (0.0 to 1.0) | float | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_out | Video with narration (path or data URI) | str (file) |
| audio_file | Generated audio file (path or data URI) | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
