
## YouTube Video Transcriber

### What it is
A specialized tool that converts the spoken content of YouTube videos into written text.

### What it does
This block takes a YouTube video URL and creates a text transcript of all spoken content in the video. It automatically detects the video's available subtitles or closed captions and converts them into readable text format.

### How it works
The block operates in three main steps:
1. It takes the YouTube URL and extracts the unique video ID
2. It searches for available transcripts/subtitles for the video
3. It converts the found transcript into a clean, readable text format

### Inputs
- YouTube URL: The web address of the YouTube video you want to transcribe. This can be in various formats, including:
  - Standard watch URLs (https://www.youtube.com/watch?v=...)
  - Short URLs (https://youtu.be/...)
  - Embed URLs (https://www.youtube.com/embed/...)

### Outputs
- Video ID: The unique identifier of the YouTube video that was processed
- Transcript: The complete text transcription of the video's spoken content, formatted as readable text
- Error: A message explaining what went wrong if the transcription process fails

### Possible use cases
- Creating written content from video lectures or educational content
- Making YouTube content accessible to deaf or hard-of-hearing viewers
- Extracting quotes or information from video interviews
- Creating searchable archives of video content
- Generating subtitles for content localization
- Research and analysis of video content
- Content repurposing (turning video content into blog posts or articles)

### Notes
- The block will automatically detect the available language for transcription
- If no transcript is available for the video, it will return an error message
- The output transcript is formatted as plain text with appropriate line breaks
- The block can handle various YouTube URL formats, making it flexible for different input styles

