# Replicate Flux Advanced
<!-- MANUAL: file_description -->
Blocks for generating images using Flux models on Replicate with advanced settings.
<!-- END MANUAL -->

## Replicate Flux Advanced Model

### What it is
This block runs Flux models on Replicate with advanced settings.

### How it works
<!-- MANUAL: how_it_works -->
The block takes a text prompt and several customization options as input. It then sends this information to the selected Flux model on the Replicate platform. The AI model processes the input and generates an image based on the provided specifications. Finally, the block returns a URL to the generated image.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | Text prompt for image generation | str | Yes |
| replicate_model_name | The name of the Image Generation Model, i.e Flux Schnell | "Flux Schnell" \| "Flux Pro" \| "Flux Pro 1.1" | No |
| seed | Random seed. Set for reproducible generation | int | No |
| steps | Number of diffusion steps | int | No |
| guidance | Controls the balance between adherence to the text prompt and image quality/diversity. Higher values make the output more closely match the prompt but may reduce overall image quality. | float | No |
| interval | Interval is a setting that increases the variance in possible outputs. Setting this value low will ensure strong prompt following with more consistent outputs. | float | No |
| aspect_ratio | Aspect ratio for the generated image | str | No |
| output_format | File format of the output image | "webp" \| "jpg" \| "png" | No |
| output_quality | Quality when saving the output images, from 0 to 100. Not relevant for .png outputs | int | No |
| safety_tolerance | Safety tolerance, 1 is most strict and 5 is most permissive | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | Generated output | str |

### Possible use case
<!-- MANUAL: use_case -->
A graphic designer could use this block to quickly generate concept art for a sci-fi game. They might input a prompt like "A futuristic spaceport on a distant planet with multiple moons in the sky" and adjust the settings to get the desired style and quality. The generated image could then serve as inspiration or a starting point for further design work.
- API Key: Your Replicate API key for authentication
- Prompt: A text description of the image you want to generate (e.g., "A futuristic cityscape at sunset")
- Image Generation Model: Choose from Flux Schnell, Flux Pro, or Flux Pro 1.1
- Seed: An optional number for reproducible image generation
- Steps: The number of diffusion steps in the image generation process
- Guidance: Controls how closely the image follows the text prompt
- Interval: Affects the variety of possible outputs
- Aspect Ratio: The width-to-height ratio of the generated image
- Output Format: Choose between WEBP, JPG, or PNG file formats
- Output Quality: Image quality setting (0-100) for JPG and WEBP formats
- Safety Tolerance: Content safety setting, from 1 (strictest) to 5 (most permissive)
<!-- END MANUAL -->

---
