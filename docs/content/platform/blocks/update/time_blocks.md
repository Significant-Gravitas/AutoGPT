
<file_name>autogpt_platform/backend/backend/blocks/compass/triggers.md</file_name>

## Compass AI Trigger

### What it is
A specialized block designed to handle and process transcription data from Compass hardware devices.

### What it does
This block receives transcription data from Compass devices and outputs the transcribed content. It processes audio transcriptions that include details such as speaker information, timing, and the actual transcribed text.

### How it works
The block operates as a webhook receiver for Compass transcription events. When a transcription is received, it extracts the transcribed content from the structured data and makes it available for further processing. The block is configured to specifically handle transcription-type webhooks from Compass devices.

### Inputs
- payload: A structured data package containing:
  - date: The date when the transcription was created
  - transcription: The complete transcribed text
  - transcriptions: A detailed list of transcription segments, each containing:
    - text: The transcribed content
    - speaker: Who was speaking
    - start: When the segment started
    - end: When the segment ended
    - duration: How long the segment lasted

### Outputs
- transcription: The complete transcribed text content from the Compass device

### Possible use cases
- Creating meeting minutes from recorded conversations
- Generating text records of customer service calls
- Documenting interviews or focus group sessions
- Real-time transcription of live presentations or speeches
- Automated note-taking during team meetings
- Creating searchable archives of spoken content

Note: This block is specifically categorized under the HARDWARE category, indicating its direct integration with physical Compass devices.
