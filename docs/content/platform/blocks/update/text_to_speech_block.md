
<file_name>autogpt_platform/backend/backend/blocks/youtube.md</file_name>

## Transcribe YouTube Video

### What it is
A specialized tool that extracts and transcribes the spoken content from YouTube videos into written text.

### What it does
This block takes a YouTube video URL as input and produces a text transcript of all spoken content in the video. It can handle various YouTube URL formats (standard watch URLs, shortened URLs, and embed URLs) and automatically identifies the video's available transcripts.

### How it works
1. Accepts a YouTube video URL from the user
2. Extracts the unique video ID from the provided URL
3. Searches for available transcripts for the video
4. Selects the first available transcript
5. Converts the transcript into readable text format
6. Returns both the video ID and the formatted transcript text

### Inputs
- YouTube URL: The web address of the YouTube video you want to transcribe. Can be in various formats, including:
  - Standard watch URLs (https://www.youtube.com/watch?v=...)
  - Shortened URLs (https://youtu.be/...)
  - Embed URLs (https://www.youtube.com/embed/...)

### Outputs
- Video ID: The unique identifier extracted from the YouTube URL
- Transcript: The complete text transcript of the video's spoken content
- Error: Any error message that might occur during the transcription process (e.g., if no transcript is available)

### Possible use cases
- Creating written documentation from educational YouTube videos
- Generating searchable text content from video lectures
- Making video content accessible to deaf or hard-of-hearing viewers
- Extracting key points from conference talks or presentations
- Converting interviews or speeches into text format for analysis
- Creating subtitles or closed captions for video content
- Research and data collection from video content

### Notes
- The block automatically handles different YouTube URL formats
- It will attempt to find available transcripts in any language
- If no transcript is available, it will return an error message
- The output text is formatted for easy reading and further processing

