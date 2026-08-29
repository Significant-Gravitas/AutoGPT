# Fal AI Video Generator
<!-- MANUAL: file_description -->
Blocks for generating AI videos using FAL.ai models.
<!-- END MANUAL -->

## AI Video Generator

### What it is
Generate videos using FAL AI models.

### How it works
<!-- MANUAL: how_it_works -->
This block generates videos from text prompts using FAL.ai's video generation models including Mochi, Luma Dream Machine, and Veo3. Describe the video you want to create, and the AI generates it.

The generated video URL is returned along with progress logs for monitoring longer generation jobs.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| prompt | Description of the video to generate. | str | Yes |
| model | The FAL model to use for video generation. | "fal-ai/mochi-v1" \| "fal-ai/luma-dream-machine" \| "fal-ai/veo3" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if video generation failed. | str |
| video_url | The URL of the generated video. | str |
| logs | Generation progress logs. | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Content Creation**: Generate video clips for social media, ads, or creative projects.

**Visualization**: Create visual representations of concepts, products, or stories.

**Prototyping**: Generate video mockups for creative ideation and storyboarding.
<!-- END MANUAL -->

---
