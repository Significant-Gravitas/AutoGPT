# Ideogram Model

## What it is
The Ideogram Model block is an AI-powered image generation tool that creates custom images based on text prompts and various settings.

## What it does
This block generates images using the Ideogram AI model, allowing users to create unique visuals by describing what they want in text. It offers various customization options, including different model versions, aspect ratios, and style preferences.

## How it works
The block takes a text prompt and several optional parameters as input. It then sends this information to the Ideogram API, which processes the request and generates an image. The resulting image URL is returned as output. If requested, the block can also upscale the generated image for higher quality.

## Inputs
| Input | Description |
|-------|-------------|
| API Key | Your personal Ideogram API key for authentication |
| Prompt | The text description of the image you want to generate |
| Image Generation Model | Choose from different versions of the Ideogram model |
| Aspect Ratio | Select the desired dimensions for your image |
| Upscale Image | Option to enhance the image quality after generation |
| Magic Prompt Option | Enables automatic enhancement of your text prompt |
| Seed | An optional number for reproducible image generation |
| Style Type | Choose a specific artistic style for your image |
| Negative Prompt | Describe elements you want to exclude from the image |
| Color Palette Preset | Select a predefined color scheme for your image |

## Outputs
| Output | Description |
|--------|-------------|
| Result | The URL of the generated image |
| Error | An error message if something goes wrong during the process |

## Possible use case
A marketing team needs unique visuals for a new product campaign. They can use the Ideogram Model block to quickly generate custom images based on their product descriptions and brand guidelines, exploring different styles and aspect ratios without the need for a professional designer.