# Compass Triggers
<!-- MANUAL: file_description -->
Blocks for triggering workflows from Compass AI transcription events.
<!-- END MANUAL -->

## Compass AI Trigger

### What it is
This block will output the contents of the compass transcription.

### How it works
<!-- MANUAL: how_it_works -->
This block triggers when a Compass AI transcription is received. It outputs the transcription text content, enabling workflows that process voice input or meeting transcripts from Compass AI.

The transcription is output as a string for downstream processing, analysis, or storage.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| transcription | The contents of the compass transcription. | str |

### Possible use case
<!-- MANUAL: use_case -->
**Voice Command Processing**: Trigger workflows from voice commands transcribed by Compass AI.

**Meeting Automation**: Process meeting transcripts to extract action items or summaries.

**Transcription Analysis**: Analyze transcribed content for sentiment, topics, or key information.
<!-- END MANUAL -->

---
