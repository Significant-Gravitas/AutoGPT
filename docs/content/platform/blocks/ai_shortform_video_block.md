# AI Shortform Video Creator

## What it is
The AI Shortform Video Creator is a tool that generates short-form videos using artificial intelligence and various customization options.

## What it does
This block creates short videos by combining AI-generated visuals, narration, and background music based on user input. It can produce different styles of videos, including stock videos, moving AI images, or AI-generated videos.

## How it works
The block takes user input for script, visual style, audio, and other parameters. It then sends this information to the revid.ai API, which processes the request and generates the video. The block monitors the video creation process and provides the final video URL once it's ready.

## Inputs
| Input | Description |
|-------|-------------|
| API Key | Your revid.ai API key for authentication |
| Script | The text content for the video, including spoken narration and visual directions |
| Ratio | The aspect ratio of the video (e.g., "9 / 16" for vertical videos) |
| Resolution | The video resolution (e.g., "720p") |
| Frame Rate | The number of frames per second in the video |
| Generation Preset | The visual style for AI-generated content (e.g., "Default", "Anime", "Realist") |
| Background Music | The choice of background music track |
| Voice | The AI voice to use for narration |
| Video Style | The type of visual media to use (stock videos, moving AI images, or AI video) |

## Outputs
| Output | Description |
|--------|-------------|
| Video URL | The web address where the created video can be accessed |
| Error | A message explaining any issues that occurred during video creation (if applicable) |

## Possible use case
A social media marketer could use this block to quickly create engaging short-form videos for platforms like TikTok or Instagram Reels. They could input a script about a new product, choose a suitable visual style and background music, and get a professional-looking video without needing advanced video editing skills.