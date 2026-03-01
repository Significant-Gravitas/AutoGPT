# Video Edit By Text
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Edit Video By Text

### What it is
Edit a video by modifying its transcript

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| video_in | Input video file to edit (URL, data URI, or local path) | str (file) | Yes |
| transcription | Desired transcript for the output video | str | Yes |
| split_at | Granularity for transcript matching | "word" \| "character" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| video_url | URL of the edited video | str |
| transcription | Transcription used for editing | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
