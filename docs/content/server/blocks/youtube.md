## Transcribe YouTube Video

### What it is
A block that transcribes the audio content of a YouTube video into text.

### What it does
This block takes a YouTube video URL as input and produces a text transcript of the video's audio content. It also extracts and provides the unique video ID associated with the YouTube video.

### How it works
The block first extracts the video ID from the provided YouTube URL. It then uses this ID to fetch the video's transcript. The transcript is processed and formatted into a readable text format. If any errors occur during this process, the block will capture and report them.

### Inputs
| Input | Description |
|-------|-------------|
| YouTube URL | The web address of the YouTube video you want to transcribe. This can be in various formats, such as a standard watch URL, a shortened URL, or an embed URL. |

### Outputs
| Output | Description |
|--------|-------------|
| Video ID | The unique identifier for the YouTube video, extracted from the input URL. |
| Transcript | The full text transcript of the video's audio content. |
| Error | Any error message that occurs if the transcription process fails. |

### Possible use case
A content creator could use this block to automatically generate subtitles for their YouTube videos. They could also use it to create text-based summaries of video content for SEO purposes or to make their content more accessible to hearing-impaired viewers.

