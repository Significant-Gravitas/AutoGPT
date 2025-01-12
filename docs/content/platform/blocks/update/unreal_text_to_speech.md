
## Unreal Text to Speech

### What it is
A specialized block that converts written text into spoken audio using the Unreal Speech API, offering high-quality text-to-speech conversion capabilities.

### What it does
This block takes written text and transforms it into natural-sounding speech, generating an audio file that can be accessed via a URL. It allows users to specify the voice they want to use for the audio generation.

### How it works
The block operates by:
1. Taking the input text and voice selection from the user
2. Sending this information to the Unreal Speech API
3. Processing the text using the specified voice settings
4. Generating an MP3 audio file
5. Providing a URL to access the generated audio

### Inputs
- Text: The written content you want to convert into speech. This can be any text that you want to hear spoken aloud.
- Voice ID: The specific voice you want to use for the speech generation. By default, it uses a voice called "Scarlett," but other voices can be selected.
- Credentials: Authentication details required to access the Unreal Speech API. This includes an API key with the necessary permissions.

### Outputs
- MP3 URL: A web address (URL) where you can access and play the generated audio file.
- Error: A message explaining what went wrong if the text-to-speech conversion fails for any reason.

### Possible use cases
- Creating audio versions of written content for accessibility purposes
- Developing voice-overs for videos or presentations
- Building interactive voice responses for applications
- Generating automated announcements or notifications
- Creating audio content for e-learning materials
- Producing spoken versions of articles or blog posts
- Developing voice-enabled assistants or chatbots
- Creating audio books from written text

### Technical Categories
- AI
- Text Processing

### Notes
- The block uses high-quality audio settings with a 192k bitrate
- The speech can be customized with additional parameters like speed and pitch
- The system supports sentence-level timestamps for precise audio timing
