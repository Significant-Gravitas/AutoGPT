
## AI Shortform Video Creator

### What it is
A powerful tool that automatically creates short-form videos by combining AI-generated or stock visuals with narration and background music.

### What it does
Transforms text scripts into engaging videos by generating visuals, adding AI voice narration, and incorporating background music. The tool can create videos in different styles, from stock footage to AI-generated content.

### How it works
1. Takes a script with special formatting where regular text is spoken and [bracketed text] guides visual generation
2. Creates a webhook to monitor the video creation process
3. Sends the request to Revid.ai with specified parameters
4. Monitors the video creation progress
5. Returns the final video URL once complete

### Inputs
- Credentials: API key required to access the Revid.ai service
- Script: Text content with special formatting (regular text for narration, [bracketed text] for visual guidance)
- Ratio: Video aspect ratio (default: 9/16 for vertical videos)
- Resolution: Video quality setting (default: 720p)
- Frame Rate: Video smoothness setting (default: 60 fps)
- Generation Preset: Visual style for AI-generated content (options include Default, Anime, Realism, Illustration, etc.)
- Background Music: Choice of background track (multiple options available like Highway Nocturne, Observer, Futuristic Beat)
- Voice: AI narrator voice selection (options include Lily, Daniel, Brian, Jessica, Charlotte, Callum)
- Video Style: Type of visuals to use (Stock Videos, Moving AI Images, or AI Video)

### Outputs
- Video URL: Direct link to the finished video
- Error: Message explaining what went wrong (if the process fails)

### Possible use cases
- Creating engaging social media content for platforms like TikTok or Instagram
- Generating product demonstrations or explanatory videos
- Producing educational content with visual aids
- Creating quick marketing videos for businesses
- Developing automated news or information snippets
- Converting blog posts or articles into video format

### Example Usage
Input script:
```
[close-up of a steaming coffee cup] Start your morning right with the perfect brew.
[barista carefully pouring latte art] Our expert baristas craft each drink with precision.
[happy customers in a cozy café] Join us for an unforgettable coffee experience.
```

This would create a video with matching visuals for each line, AI narration, and background music, perfect for a café's social media marketing.

