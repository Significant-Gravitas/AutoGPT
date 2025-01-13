
<file_name>autogpt_platform/backend/backend/blocks/ideogram.md</file_name>

## Ideogram Model Block

### What it is
A specialized block that interfaces with Ideogram's AI image generation models to create custom images based on text descriptions.

### What it does
Generates images from text prompts using Ideogram's AI models, with options for different styles, aspect ratios, and image enhancement features.

### How it works
The block takes a text prompt and various customization parameters, sends them to the Ideogram API, and returns a URL for the generated image. It can optionally upscale the generated image for higher quality results.

### Inputs
- Credentials: Ideogram API key required for authentication
- Prompt: Text description of the image you want to generate
- Image Generation Model: Choice of Ideogram model version (V1, V2, V1 Turbo, V2 Turbo)
- Aspect Ratio: Image dimensions ratio (various options from square to landscape/portrait)
- Upscale Image: Option to enhance image quality using AI upscaling
- Magic Prompt Option: Setting to automatically enhance the input prompt
- Seed: Optional number for reproducible image generation
- Style Type: Visual style preference (General, Realistic, Design, 3D Render, Anime)
- Negative Prompt: Description of elements to exclude from the image
- Color Palette Preset: Predefined color schemes (Ember, Fresh, Jungle, etc.)

### Outputs
- Result: URL of the generated image
- Error: Error message if the generation process fails

### Possible use cases
- Creating custom illustrations for articles or blog posts
- Generating concept art for creative projects
- Designing marketing materials with specific visual requirements
- Producing consistent brand imagery with predefined color palettes
- Creating social media content with specific aspect ratios
- Developing visual prototypes for product design
- Generating mood boards with consistent artistic styles
- Creating custom backgrounds for presentations or websites

### Advanced Features
The block includes several advanced customization options:
- Reproducible generations using seed values
- Style control for specific artistic looks
- Negative prompting for fine-tuned results
- Color palette presets for consistent branding
- AI upscaling for higher resolution outputs
- Magic prompt enhancement for better results

