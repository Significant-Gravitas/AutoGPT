# Replicate Replicate Block
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Replicate Model

### What it is
Run Replicate models synchronously

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| model_name | The Replicate model name (format: 'owner/model-name') | str | Yes |
| model_inputs | Dictionary of inputs to pass to the model. Values may be strings, integers, floats, or booleans — Replicate model schemas commonly require booleans (e.g. ``generate_audio``, ``safety_checker``) and floats (e.g. ``temperature``, ``guidance_scale``). | Dict[str, str \| int \| float \| bool] | No |
| files | Files (image, audio, video, etc.) to send to the model. Each file is uploaded to Replicate and passed to the model under the field named by ``file_input_field``. Pass file references (URLs, uploaded files) here instead of inlining base64 in ``model_inputs``. | List[str (file)] | No |
| file_input_field | Name of the model input field that receives the uploaded ``files``. Defaults to ``image``, the most common name. Check the model's schema on Replicate for the exact name (common alternatives: ``input_image``, ``img``, ``image_input``, ``audio``, ``video``, ``mask``). A single file is sent as one value; multiple files are sent as a list. | str | No |
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
