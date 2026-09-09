# Gandr Text To Speech
<!-- MANUAL: file_description -->
Block for converting text into spoken audio using the Gandr text to speech API.
<!-- END MANUAL -->

## Gandr Text To Speech

### What it is
Converts text to speech using the Gandr API

### How it works
<!-- MANUAL: how_it_works -->
The block sends the text and voice selection to the Gandr speech endpoint. The API returns an MP3 render, which the block stores and outputs as a media file. Gandr covers 23 languages with six stock voices. Every render is watermarked.

Each request accepts up to 2000 characters. Longer text should be split and run once per chunk. API keys come from [https://gandr.ai](https://gandr.ai) and the free tier is 50,000 tokens.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| text | The text to convert to speech. Up to 2000 characters per request. | str | Yes |
| voice | The Gandr voice to use | "gandr-mia" \| "gandr-ava" \| "gandr-jenny" \| "gandr-dane" \| "gandr-leo" \| "gandr-lewis" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| audio_file | Generated MP3 audio (path or data URI) | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
**Voiceover for generated video**: Feed script text into the block and pass the audio file to a video assembly block.

**Spoken alerts**: Convert workflow results into short spoken updates.

**Multilingual narration**: Render a script in any of the 23 supported languages by writing the text in that language.
<!-- END MANUAL -->

---
