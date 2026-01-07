# Replicate Model

### What it is
Run Replicate models synchronously.

### What it does
Run Replicate models synchronously

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| model_name | The Replicate model name (format: 'owner/model-name') | str | Yes |
| model_inputs | Dictionary of inputs to pass to the model | Dict[str, str | int] | No |
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
