
# Text to Speech Blocks Documentation

## Unreal Text to Speech

### What it is
A specialized block that converts written text into spoken audio using the Unreal Speech API service.

### What it does
This block takes text input and transforms it into natural-sounding speech, creating an audio file that can be accessed via a URL. It utilizes Unreal Speech's advanced text-to-speech technology to generate high-quality voice output.

### How it works
1. The block receives text input and voice preferences from the user
2. It securely connects to the Unreal Speech API using provided credentials
3. The text is processed and converted to speech using the specified voice
4. The API generates an MP3 audio file and returns its URL
5. The block provides the URL where the audio file can be accessed

### Inputs
- Text: The written content you want to convert into speech (required)
- Voice ID: The specific voice you want to use for the speech generation (defaults to "Scarlett")
- Credentials: API authentication details needed to access the Unreal Speech service

### Outputs
- MP3 URL: A web link to access the generated audio file
- Error: A message explaining what went wrong if the conversion fails

### Possible use cases
- Creating voiceovers for videos or presentations
- Generating audio versions of written content for accessibility
- Developing audio-based learning materials
- Creating voice prompts for applications or systems
- Converting articles or books into audio format for listening
- Generating automated announcements or notifications
- Creating audio content for podcasts or broadcasting

### Additional Notes
- The block supports customizable voice options through the Voice ID parameter
- Audio is generated in MP3 format with high-quality 192k bitrate
- The system includes built-in error handling to ensure reliable operation
- Voice speed and pitch can be adjusted through the API
- The service includes sentence-level timestamp capabilities
- The block is categorized under both AI and Text processing capabilities

