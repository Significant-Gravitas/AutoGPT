## Edit Video by Text

### What it is
A block that edits a video by cutting segments based on an edited transcript.

### What it does
After providing a target transcript, the block removes portions of the video that no longer appear in the text, returning a new edited video file.

### How it works
The block compares the supplied transcript with the video's original transcript. Segments that are missing from the target transcript are removed. Word-level matching is used by default.

### Inputs
| Input | Description |
|-------|-------------|
| Video | The original video file to edit. |
| Transcription | The desired transcript of the output video. |
| Split At | Level of precision for transcript matching ("word" or "character"). |

### Outputs
| Output | Description |
|--------|-------------|
| Video | Path to the edited video. |
| Transcription | The transcript used to generate the edited video. |
| Error | Error message if editing fails. |

### Possible use case
Create a shorter version of a training video by removing sentences from the transcript instead of using a timeline-based video editor.
