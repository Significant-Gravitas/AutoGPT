# Slant3D Slicing
<!-- MANUAL: file_description -->
Blocks for slicing 3D models and getting pricing information from Slant3D.
<!-- END MANUAL -->

## Slant3D Slicer

### What it is
Slice a 3D model file and get pricing information

### How it works
<!-- MANUAL: how_it_works -->
This block sends an STL file to Slant3D's slicing service to analyze the 3D model. The slicer calculates print parameters and returns pricing information based on the model's complexity, size, and material requirements.

Provide the URL to your STL file, and the block returns the calculated price for printing the model.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file_url | URL of the 3D model file to slice (STL) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message | Response message | str |
| price | Calculated price for printing | float |

### Possible use case
<!-- MANUAL: use_case -->
**Price Quoting**: Get instant price quotes for 3D models uploaded by customers.

**Model Validation**: Verify that STL files are printable before accepting orders.

**Cost Estimation**: Calculate printing costs as part of an automated quoting system.
<!-- END MANUAL -->

---
