
## Unreal Text to Speech

### What it is
A specialized block that converts written text into natural-sounding speech using the Unreal Speech API technology.

### What it does
This block takes written text and transforms it into spoken audio, creating an MP3 file that can be accessed via a URL. It uses advanced AI voice technology to generate human-like speech with customizable voice options.

### How it works
The block processes your input text and selected voice preference, sends this information to the Unreal Speech API service, and returns a link to an MP3 audio file containing the generated speech. The process happens in real-time, and the resulting audio file can be easily accessed through the provided URL.

### Inputs
- Text: The written content you want to convert into speech. This can be any text content, from single sentences to longer paragraphs.
- Voice ID: The specific voice you want to use for the speech generation. By default, it uses a voice called "Scarlett," but other voices can be selected.
- Credentials: Your Unreal Speech API authentication details. This is required to access the service and is managed through an API key system.

### Outputs
- MP3 URL: A web link to the generated audio file in MP3 format. This file contains your text converted to speech and can be played or downloaded.
- Error: If something goes wrong during the process, this will contain a message explaining what happened. This helps users understand and troubleshoot any issues that might occur.

### Possible use cases
- Creating audio versions of written content for accessibility purposes
- Developing automated voice responses for customer service systems
- Generating voiceovers for educational content or presentations
- Creating audio versions of articles or blog posts for podcast-style consumption
- Building interactive voice applications or voice-enabled features in applications
- Creating audio books from digital text
- Developing learning materials for language education

### Notes
- The block is categorized under both AI and Text processing capabilities
- The speech generation includes options for voice customization
- The service requires valid API credentials to function
- The generated audio is delivered in high-quality MP3 format (192k bitrate)

