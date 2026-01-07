# Bannerbear Text Overlay

### What it is
Add text overlay to images using Bannerbear templates.

### What it does
Add text overlay to images using Bannerbear templates. Perfect for creating social media graphics, marketing materials, and dynamic image content.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| template_id | The unique ID of your Bannerbear template | str | Yes |
| project_id | Optional: Project ID (required when using Master API Key) | str | No |
| text_modifications | List of text layers to modify in the template | List[TextModification] | Yes |
| image_url | Optional: URL of an image to use in the template | str | No |
| image_layer_name | Optional: Name of the image layer in the template | str | No |
| webhook_url | Optional: URL to receive webhook notification when image is ready | str | No |
| metadata | Optional: Custom metadata to attach to the image | str | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| success | Whether the image generation was successfully initiated | bool |
| image_url | URL of the generated image (if synchronous) or placeholder | str |
| uid | Unique identifier for the generated image | str |
| status | Status of the image generation | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
