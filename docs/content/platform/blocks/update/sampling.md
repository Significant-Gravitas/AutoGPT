
## Create Talking Avatar Video

### What it is
A specialized block that creates personalized video content featuring an AI-powered talking avatar using the D-ID platform.

### What it does
This block transforms text input into a video featuring a digital presenter who speaks the provided script. It can create videos with different voices, presenters, and formatting options, with the ability to include subtitles if desired.

### How it works
The block takes your input text and configuration options, sends them to the D-ID service, and monitors the video creation process. Once the video is ready, it provides you with a URL where you can access the finished video.

### Inputs
- Credentials: D-ID API key required for accessing the service
- Script Input: The text that the avatar will speak in the video
- Voice Provider: Choice of voice service (Microsoft, ElevenLabs, or Amazon)
- Voice ID: Specific voice identifier for the chosen provider
- Presenter ID: The digital avatar that will appear in the video
- Driver ID: Controls how the avatar animates
- Result Format: Output format of the video (MP4, GIF, or WAV)
- Crop Type: Video frame format (wide, square, or vertical)
- Subtitles: Option to include subtitles in the video
- SSML: Option to use Speech Synthesis Markup Language for more control over speech
- Max Polling Attempts: Number of times to check if the video is ready
- Polling Interval: Time to wait between checking video status

### Outputs
- Video URL: Web address where the completed video can be accessed
- Error: Message explaining what went wrong if the video creation fails

### Possible use case
A marketing team wants to create personalized welcome videos for new customers. They could use this block to generate videos where a digital presenter greets each customer by name and explains the company's key features. The team could select a professional-looking avatar, choose an appropriate voice, and even include subtitles for accessibility.

For example:
1. Input a script: "Welcome, [Customer Name]! We're excited to have you join us..."
2. Select a professional presenter avatar
3. Choose a natural-sounding voice
4. Enable subtitles for accessibility
5. Get back a video URL that can be sent to the customer

This creates a scalable way to deliver personalized video content without needing to record individual videos for each customer.

