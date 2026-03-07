# Video Edit By Text
<!-- MANUAL: file_description -->
This block edits a video by modifying its transcript — segments absent from the supplied transcript are cut from the output video, powered by the Replicate API.
<!-- END MANUAL -->

## Edit Video By Text

### What it is
Edit a video by modifying its transcript

### What it does
Takes a video and a modified version of its transcript, then produces a new video with only the segments that match the provided transcript. Any spoken segments you remove from the transcript will be cut from the output video.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the input video and the desired transcript to the Replicate API using the `jd7h/edit-video-by-editing-text` model in "edit" mode. The model aligns the provided transcript against the original speech in the video and removes any video segments whose speech is not present in the supplied transcript. The `split_at` parameter controls alignment granularity: `word` (default) aligns cuts at word boundaries for natural-sounding edits, while `character` allows finer sub-word alignment for more precise control. The block returns the URL of the edited video along with the transcript that was used.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| video_in | Input video file to edit (URL, data URI, or local path) | str (file) | Yes |
| transcription | Modified transcript of the input video — segments absent from this text will be cut from the output video | str | Yes |
| split_at | Alignment granularity for transcript matching: `word` aligns cuts at word boundaries (default), `character` allows finer sub-word alignment | "word" \| "character" | No (default: `word`) |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_url | URL of the edited video | str |
| transcription | Transcription used for editing | str |

### Possible use case
<!-- MANUAL: use_case -->
**Interview Cleanup**: Remove filler words, false starts, or off-topic tangents from recorded interviews by editing the transcript and regenerating the video.

**Content Highlights**: Extract key segments from long-form video content by keeping only the relevant portions of the transcript.

**Automated Moderation**: Remove flagged or inappropriate speech segments from user-generated video content by stripping those lines from the transcript.
<!-- END MANUAL -->

---
