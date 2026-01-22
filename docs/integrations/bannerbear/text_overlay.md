# Bannerbear Text Overlay
<!-- MANUAL: file_description -->
Blocks for generating dynamic images with text overlays using Bannerbear templates.
<!-- END MANUAL -->

## Bannerbear Text Overlay

### What it is
Add text overlay to images using Bannerbear templates. Perfect for creating social media graphics, marketing materials, and dynamic image content.

### How it works
<!-- MANUAL: how_it_works -->
This block uses Bannerbear's API to generate images by populating templates with dynamic text and images. Create templates in Bannerbear with text layers, then modify layer content programmatically.

Webhooks can notify you when asynchronous generation completes. Include custom metadata for tracking generated images.
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
**Social Media Graphics**: Generate personalized social posts with dynamic quotes, stats, or headlines.

**Marketing Banners**: Create ad banners with different product names, prices, or offers.

**Certificates & Cards**: Generate personalized certificates, invitations, or greeting cards.
<!-- END MANUAL -->

---
