# Wavespeed AI Media Generator
<!-- MANUAL: file_description -->
Blocks for generating AI images and videos using WaveSpeed models.
<!-- END MANUAL -->

## AI Media Generator

### What it is
Generate images and videos using WaveSpeed AI models.

### How it works
<!-- MANUAL: how_it_works -->
This block runs any image or video generation model on the live WaveSpeed catalog, including Seedream, Seedance, FLUX, and Wan. Pick a model ID from wavespeed.ai, describe what you want to generate, and optionally pass model-specific parameters (size, seed, an input image URL, ...) via the extra inputs field.

The block submits the generation request, polls until the prediction completes, and returns the generated media URL(s).
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| model | The WaveSpeed model ID (format: 'owner/model-name'). See https://wavespeed.ai for the full catalog. | str | No |
| prompt | Text prompt describing the image or video to generate. | str | Yes |
| extra_inputs | Additional model-specific inputs to include in the request body, e.g. size, seed, or an image URL for image-to-image / image-to-video models. Check the model's page on wavespeed.ai for its input schema. | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if generation failed. | str |
| media_url | The URL of the (first) generated image or video. | str |
| media_urls | URLs of all generated outputs. | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
Generate product imagery with Seedream for a marketing agent, or turn a still image into a short video clip with Seedance as part of a content pipeline.
<!-- END MANUAL -->

---
