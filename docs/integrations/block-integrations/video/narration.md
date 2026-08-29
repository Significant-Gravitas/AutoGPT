# Video Narration
<!-- MANUAL: file_description -->
This block generates AI voiceover narration using ElevenLabs and adds it to a video, with flexible audio mixing options.
<!-- END MANUAL -->

## Video Narration

### What it is
Generate AI narration and add to video

### How it works
<!-- MANUAL: how_it_works -->
The block uses ElevenLabs text-to-speech API to generate natural-sounding narration from your script. It then combines the narration with the video using MoviePy. Three audio mixing modes are available: **replace** (completely replaces original audio), **mix** (blends narration with original audio at configurable volumes), and **ducking** (similar to mix but applies stronger attenuation to original audio, making narration more prominent). The block outputs both the final video and the generated audio file separately.
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
- Adding professional voiceover to product demos or tutorials
- Creating narrated explainer videos from screen recordings
- Generating multi-language versions of video content
- Adding commentary to gameplay or walkthrough videos
<!-- END MANUAL -->

---
