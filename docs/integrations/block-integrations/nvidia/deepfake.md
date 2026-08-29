# Nvidia Deepfake
<!-- MANUAL: file_description -->
Blocks for detecting deepfakes and synthetic image manipulation using Nvidia AI.
<!-- END MANUAL -->

## Nvidia Deepfake Detect

### What it is
Detects potential deepfakes in images using Nvidia's AI API

### How it works
<!-- MANUAL: how_it_works -->
This block analyzes images using Nvidia's AI-powered deepfake detection model. It returns a probability score (0-1) indicating the likelihood that an image has been synthetically manipulated.

Set return_image to true to receive a processed image with detection markings highlighting areas of concern.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| image_base64 | Image to analyze for deepfakes | str (file) | Yes |
| return_image | Whether to return the processed image with markings | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | Detection status (SUCCESS, ERROR, CONTENT_FILTERED) | str |
| image | Processed image with detection markings (if return_image=True) | str (file) |
| is_deepfake | Probability that the image is a deepfake (0-1) | float |

### Possible use case
<!-- MANUAL: use_case -->
**Content Verification**: Verify authenticity of user-uploaded profile photos or identity documents.

**Media Integrity**: Screen submitted images for signs of AI manipulation.

**Trust & Safety**: Detect potentially misleading synthetic content in social or news platforms.
<!-- END MANUAL -->

---
