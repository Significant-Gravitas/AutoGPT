
## AI Shortform Video Creator

### What it is
A powerful tool that creates short-form videos using AI technology through the revid.ai service. This block combines text, visuals, voice narration, and music to generate engaging video content.

### What it does
Transforms written scripts into complete videos by:
- Converting text to AI-generated voice narration
- Adding visual elements based on text descriptions
- Including background music
- Combining all elements into a cohesive video with specified dimensions and quality

### How it works
1. Takes a script with spoken text and visual descriptions
2. Creates a webhook to monitor the video creation process
3. Sends the creation request to revid.ai with all specified parameters
4. Monitors the creation process until completion
5. Returns the final video URL once ready

### Inputs
- Credentials: API key credentials for accessing the revid.ai service
- Script: Text content with spoken dialogue and visual descriptions in brackets
- Ratio: Video aspect ratio (default: 9/16 for vertical videos)
- Resolution: Video quality setting (default: 720p)
- Frame Rate: Video smoothness setting (default: 60 fps)
- Generation Preset: Visual style for AI-generated content (options include Default, Anime, Realism, etc.)
- Background Music: Choice of music track from available options
- Voice: Selection of AI voice for narration (options include Lily, Daniel, Brian, etc.)
- Video Style: Type of visuals to use (Stock Videos, Moving AI Images, or AI Video)

### Outputs
- Video URL: Direct link to the completed video
- Error: Description of any issues that occurred during video creation

### Possible use cases
- Creating social media content for platforms like TikTok or Instagram
- Generating educational content with visual demonstrations
- Producing product demonstrations with narration and visuals
- Creating engaging marketing videos with minimal effort
- Developing automated content for social media campaigns
- Producing quick explainer videos with professional narration

Example Script:
"[close-up of a coffee cup] Start your morning right with the perfect cup of coffee. [barista pouring latte art] Our expert baristas craft each drink with precision and care."

This block is particularly useful for content creators, marketers, and businesses looking to create professional-quality short-form videos without extensive video production expertise or resources.

