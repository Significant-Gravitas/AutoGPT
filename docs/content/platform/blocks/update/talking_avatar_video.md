
## Create Talking Avatar Video

### What it is
A specialized block that creates video clips featuring an AI-powered digital avatar that speaks your provided text using various voice options.

### What it does
This block transforms text input into a video featuring a digital presenter speaking the provided content. It connects with D-ID's service to generate realistic talking avatar videos with customizable voices and presentation styles.

### How it works
1. Takes your text input and voice preferences
2. Sends the information to D-ID's service
3. Creates a video clip with a digital presenter speaking your text
4. Monitors the creation process until completion
5. Provides you with the final video URL

### Inputs
- Credentials: Your D-ID API key for accessing the service
- Script Input: The text you want the avatar to speak
- Provider: Choice of voice provider (Microsoft, ElevenLabs, or Amazon)
- Voice ID: The specific voice to use for speaking
- Presenter ID: The digital avatar to use in the video
- Driver ID: The animation style for the presenter
- Result Format: Output format (MP4, GIF, or WAV)
- Crop Type: Video frame style (wide, square, or vertical)
- Subtitles: Option to include subtitles in the video
- SSML: Option to use Speech Synthesis Markup Language for advanced voice control
- Max Polling Attempts: Maximum number of times to check if the video is ready
- Polling Interval: Time to wait between checking video status

### Outputs
- Video URL: The web address where you can access your completed video
- Error: Any error message if the video creation fails

### Possible use cases
- Creating engaging video presentations for social media
- Generating multilingual training materials with consistent presenters
- Producing educational content with a virtual instructor
- Creating personalized welcome messages for websites
- Developing interactive customer service responses
- Making product demonstrations with consistent branding
- Creating automated news broadcasts
- Developing e-learning materials with visual engagement

**Note:** To ensure successful video creation, make sure to use valid voice IDs and presenter IDs as specified in the D-ID documentation. The process may take several minutes depending on the length of your script and server load.

