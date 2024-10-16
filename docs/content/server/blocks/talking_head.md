## Create Talking Avatar Video

### What it is
This block is an AI-powered tool that creates video clips featuring a talking avatar using the D-ID service.

### What it does
It generates a video of a digital avatar speaking a given script, with customizable voice, presenter, and visual settings.

### How it works
The block sends a request to the D-ID API with your specified parameters. It then regularly checks the status of the video creation process until it's complete or an error occurs.

### Inputs
| Input | Description |
|-------|-------------|
| API Key | Your D-ID API key for authentication |
| Script Input | The text you want the avatar to speak |
| Provider | The voice provider to use (options: microsoft, elevenlabs, amazon) |
| Voice ID | The specific voice to use for the avatar |
| Presenter ID | The visual appearance of the avatar |
| Driver ID | The animation style for the avatar |
| Result Format | The file format of the final video (options: mp4, gif, wav) |
| Crop Type | How the video should be cropped (options: wide, square, vertical) |
| Subtitles | Whether to include subtitles in the video |
| SSML | Whether the input script uses Speech Synthesis Markup Language |
| Max Polling Attempts | Maximum number of times to check for video completion |
| Polling Interval | Time to wait between each status check (in seconds) |

### Outputs
| Output | Description |
|--------|-------------|
| Video URL | The web address where you can access the completed video |
| Error | A message explaining what went wrong if the video creation failed |

### Possible use case
A marketing team could use this block to create engaging video content for social media. They could input a script promoting a new product, select a friendly-looking avatar, and generate a video that explains the product's features in an appealing way.