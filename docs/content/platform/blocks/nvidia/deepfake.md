# Nvidia Deepfake Detect

### What it is
Detects potential deepfakes in images using Nvidia's AI API.

### What it does
Detects potential deepfakes in images using Nvidia's AI API

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
