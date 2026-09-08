# Slant3D Slicing
<!-- MANUAL: file_description -->
Blocks for slicing 3D models and getting pricing information from Slant3D.
<!-- END MANUAL -->

## Slant3D Slicer

### What it is
Upload or reuse an STL file and estimate its printing cost

### How it works
<!-- MANUAL: how_it_works -->
This block uploads a public STL URL using Slant3D's signed upload flow, confirms the upload, and requests a printing estimate. Supply file_id to reuse an existing upload. Set platform_id, or omit it when the account has exactly one enabled platform.

The price is in USD for the requested quantity and excludes shipping. filament_id selects a public filament ID; omitting it uses Slant3D's default black PLA. The returned file_id can be reused in order items.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file_url | Public STL URL to upload; ignored when file_id is set | str | No |
| file_id | Previously uploaded Slant3D public file service ID | str | No |
| platform_id | Platform ID for uploads; may be omitted with one enabled platform | str | No |
| filament_id | Filament public ID; defaults to Slant3D's PLA Black | str | No |
| quantity | Number of prints to estimate | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| message | Response message | str |
| price | Estimated printing price for the requested quantity in USD | float |
| file_id | Slant3D public file service ID for order items | str |

### Possible use case
<!-- MANUAL: use_case -->
**Price Quoting**: Get instant price quotes for 3D models uploaded by customers.

**Model Validation**: Verify that STL files are printable before accepting orders.

**Cost Estimation**: Calculate printing costs as part of an automated quoting system.
<!-- END MANUAL -->

---
