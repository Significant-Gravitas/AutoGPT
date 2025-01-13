
<file_name>autogpt_platform/backend/backend/blocks/talking_head.md</file_name>

## Create Talking Avatar Video

### What it is
A specialized block that creates AI-powered talking avatar videos using the D-ID platform, combining visual avatars with voice synthesis.

### What it does
Generates video clips featuring a digital presenter speaking provided text content, with customizable voice, presenter appearance, and video format options.

### How it works
1. Takes user input including script text and configuration options
2. Sends the request to D-ID's API to create a video clip
3. Monitors the creation process until completion
4. Provides the URL of the finished video

### Inputs
- Credentials: D-ID API key for authentication
- Script Input: The text content that the avatar will speak
- Provider: Voice provider selection (Microsoft, ElevenLabs, or Amazon)
- Voice ID: Specific voice identifier for the chosen provider
- Presenter ID: The digital avatar to use in the video
- Driver ID: Animation controller for the avatar's movements
- Result Format: Output format (MP4, GIF, or WAV)
- Crop Type: Video frame composition (wide, square, or vertical)
- Subtitles: Option to enable or disable subtitles
- SSML: Option to use Speech Synthesis Markup Language for advanced voice control
- Max Polling Attempts: Maximum number of status checks
- Polling Interval: Time between status checks in seconds

### Outputs
- Video URL: Direct link to the generated video
- Error: Description of any issues that occurred during video creation

### Possible use cases
- Creating AI-powered video presentations
- Generating multilingual training materials with consistent presenters
- Producing automated welcome messages for websites
- Creating educational content with engaging visual delivery
- Developing personalized video messages for marketing campaigns
- Building interactive virtual guides or tour hosts

### Notes
- The block requires valid D-ID API credentials
- Video generation time may vary based on content length and complexity
- Supports multiple voice providers for flexibility in speech synthesis
- Offers various output formats suitable for different platforms and uses
- Includes customizable polling parameters for processing longer videos

