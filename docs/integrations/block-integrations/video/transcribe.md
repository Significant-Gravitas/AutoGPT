# Video Transcribe
<!-- MANUAL: file_description -->
This block transcribes speech from a video file to text using the Replicate API.
<!-- END MANUAL -->

## Transcribe Video

### What it is
Extract spoken words from a video and return them as a text transcription

### How it works
<!-- MANUAL: how_it_works -->
The block sends the input video to the Replicate API using the `jd7h/edit-video-by-editing-text` model in "transcribe" mode. This model analyzes the audio track of the video, performs speech recognition, and returns the detected speech as text. The block handles multiple API response formats (dictionary, list, string, and file output) to reliably extract the transcript text.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| video_in | Input video file to transcribe (URL, data URI, or local path) | str (file) | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| transcription | Text transcription extracted from the video | str |

### Possible use case
<!-- MANUAL: use_case -->
**Subtitle Generation**: Transcribe video dialogue to create subtitle or caption files for accessibility and localization.

**Searchable Video Archives**: Convert speech in recorded meetings, interviews, or lectures into searchable text for indexing and retrieval.

**LLM Content Pipeline**: Feed video transcripts into language models for summarization, analysis, or content repurposing workflows.
<!-- END MANUAL -->

---
