# Replicate Flux Advanced
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Replicate Flux Advanced Model

### What it is
This block runs Flux models on Replicate with advanced settings.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
