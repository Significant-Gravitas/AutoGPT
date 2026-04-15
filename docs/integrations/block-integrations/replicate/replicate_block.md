# Replicate Replicate Block
<!-- MANUAL: file_description -->
Blocks for running any AI model hosted on the Replicate platform.
<!-- END MANUAL -->

## Replicate Model

### What it is
Run Replicate models synchronously

### How it works
<!-- MANUAL: how_it_works -->
This block runs any model hosted on Replicate using their API. Specify the model name in owner/model format, provide inputs as a dictionary, and optionally pin to a specific version.

The block waits for completion and returns the model output along with status information.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| model_name | The Replicate model name (format: 'owner/model-name') | str | Yes |
| model_inputs | Dictionary of inputs to pass to the model | Dict[str, str \| int] | No |
| version | Specific version hash of the model (optional) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | The output from the Replicate model | str |
| status | Status of the prediction | str |
| model_name | Name of the model used | str |

### Possible use case
<!-- MANUAL: use_case -->
**Model Flexibility**: Access thousands of open-source AI models from a single interface.

**Custom Models**: Run your own models deployed on Replicate in workflows.

**Specialized AI Tasks**: Use best-of-breed models for specific tasks like upscaling, segmentation, or captioning.
<!-- END MANUAL -->

---
