
## AI Shortform Video Creator

### What it is
A powerful tool that creates short-form videos using AI technology, combining visual elements, voice narration, and background music.

### What it does
This block transforms text scripts into engaging short-form videos by automatically generating visuals, adding AI voice narration, and incorporating background music. It supports various visual styles, aspect ratios, and customization options.

### How it works
1. Takes a script with special formatting where regular text is spoken and text in brackets guides visual generation
2. Creates a webhook to monitor the video creation process
3. Submits the video creation request with specified parameters
4. Monitors the creation process until completion
5. Returns the final video URL when ready

### Inputs
- Credentials: API key credentials for accessing the revid.ai service
- Script: Text content with special formatting where regular text is spoken and [bracketed text] guides visual generation
- Ratio: Video aspect ratio (default: "9/16" for vertical videos)
- Resolution: Video quality setting (default: "720p")
- Frame Rate: Video smoothness setting (default: 60 fps)
- Generation Preset: Visual style for AI-generated content, with options including:
  - Default
  - Anime
  - Realism
  - Illustration
  - Sketch (Color and B&W)
  - Pixar
  - Japanese Ink
  - 3D Render
  - And many more styles
- Background Music: Choice of background track from a curated selection
- Voice: AI narrator voice selection (options include Lily, Daniel, Brian, Jessica, Charlotte, and Callum)
- Video Style: Type of visual media (Stock Videos, Moving AI Images, or AI Video)

### Outputs
- Video URL: Direct link to the completed video
- Error: Message explaining any issues that occurred during video creation

### Possible use cases
- Creating engaging social media content for platforms like TikTok or Instagram
- Producing educational content with AI-generated visuals and professional narration
- Making product demonstrations with custom visual styles
- Generating quick marketing videos with consistent branding
- Creating storytelling content with synchronized visuals and narration
- Producing automated news summaries with relevant visuals

### Note
The system is designed to be user-friendly while offering professional-grade video creation capabilities. Users can focus on their content while the AI handles the technical aspects of video production, including visual generation, voice synthesis, and audio-visual synchronization.
