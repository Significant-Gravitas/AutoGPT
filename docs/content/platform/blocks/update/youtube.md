
## Transcribe YouTube Video

### What it is
A tool that automatically extracts and converts spoken content from YouTube videos into written text.

### What it does
This block takes a YouTube video URL and creates a text transcript of all spoken content in the video. It can handle various YouTube URL formats (standard watch URLs, shortened URLs, and embedded URLs) and produces a readable text version of the video's audio content.

### How it works
The block operates in three main steps:
1. It takes the provided YouTube URL and extracts the unique video ID
2. It searches for available transcripts for the video
3. It formats the transcript into readable text, maintaining the original sequence of spoken content

### Inputs
- YouTube URL: The web address of the YouTube video you want to transcribe. This can be in various formats:
  - Standard format (https://www.youtube.com/watch?v=...)
  - Shortened format (https://youtu.be/...)
  - Embedded format (https://www.youtube.com/embed/...)

### Outputs
- Video ID: The unique identifier of the YouTube video that was processed
- Transcript: The complete text version of the video's spoken content, formatted for easy reading
- Error: A message explaining what went wrong if the transcription process fails (for example, if no transcript is available for the video)

### Possible use cases
- Creating written documentation from educational YouTube videos
- Making video content accessible to deaf or hard-of-hearing viewers
- Generating searchable text content from video presentations
- Creating subtitles or closed captions for video content
- Extracting quotes or specific segments from video interviews
- Research and analysis of video content through text-based processing

### Notes
- The block will attempt to find transcripts in any available language if the primary language isn't available
- Not all YouTube videos have available transcripts, in which case the block will return an error message
- The quality of the transcript depends on the original caption quality provided by YouTube
