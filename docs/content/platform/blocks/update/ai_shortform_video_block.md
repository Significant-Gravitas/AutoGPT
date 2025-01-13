
<file_name>autogpt_platform/backend/backend/blocks/ai_shortform_video_block.md</file_name>

## AI Shortform Video Creator

### What it is
A specialized block that creates short-form videos using the revid.ai platform, combining AI-generated visuals, voice narration, and background music.

### What it does
Generates engaging short-form videos by combining user-provided scripts with AI-powered visuals, professional voiceovers, and background music. It supports various visual styles, aspect ratios, and customization options.

### How it works
1. Takes a script with visual instructions and converts it into video segments
2. Processes the script using AI to generate or select appropriate visuals
3. Adds selected AI voice narration to the content
4. Incorporates chosen background music
5. Combines all elements into a final video with specified parameters
6. Delivers a URL to the completed video

### Inputs
- Credentials: API key for accessing the revid.ai service
- Script: Text content with visual instructions in brackets (e.g., "[close-up of a cat] Meow!")
- Ratio: Video aspect ratio (default: 9/16 for vertical videos)
- Resolution: Video quality setting (default: 720p)
- Frame Rate: Video smoothness setting (default: 60 fps)
- Generation Preset: Visual style for AI-generated content (options include Default, Anime, Realism, etc.)
- Background Music: Choice of audio track from predefined options
- Voice: Selection of AI voice for narration (options include Lily, Daniel, Brian, etc.)
- Video Style: Type of visuals to use (Stock Videos, Moving AI Images, or AI Video)

### Outputs
- Video URL: Direct link to the generated video
- Error: Error message if video creation fails

### Possible use cases
- Creating engaging social media content for platforms like TikTok or Instagram
- Producing educational content with visual explanations
- Generating product demonstrations with AI-powered visuals
- Creating quick storytelling videos for marketing campaigns
- Developing automated content for social media channels
- Producing explainer videos with professional narration

### Features
- Multiple visual style presets for different aesthetic requirements
- Professional AI voice options for narration
- Diverse background music selection
- Support for various aspect ratios and resolutions
- Flexible script formatting with visual instructions
- Real-time video generation status monitoring
- Webhook integration for status updates

The block particularly excels in automating video creation while maintaining professional quality and customization options, making it ideal for content creators, marketers, and educators who need to produce engaging short-form video content efficiently.

