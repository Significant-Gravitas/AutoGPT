# Slant3D Slicing
<!-- MANUAL: file_description -->
Blocks for slicing 3D models and getting pricing information from Slant3D.
<!-- END MANUAL -->

## Slant3D Slicer

### What it is
Get a live 3D printing quote for physical parts from STL URLs or attached workspace files. Use this to price printable parts and complete project part sets before ordering from Slant3D, which manufactures and ships the parts. Binary STL URLs are accepted directly; no browser upload is needed. For complete projects, first read the assembly bill of materials: include repeated parts across subassemblies, select a design variant for alternative parts while keeping the required number of copies, and exclude reference assembly meshes. One copy of each STL is not necessarily a complete set. Quote each required file with its full assembly quantity and sum the returned prices, which already include quantity. Before reporting a complete quote, reconcile all quoted files and quantities against the parts list. Printing-only quotes need no shipping address or payment method. Returns a reusable file_id for ordering; this block does not place or charge an order.

### How it works
<!-- MANUAL: how_it_works -->
This block loads an STL URL, workspace attachment, or data URI through the shared media loader, then uploads it using Slant3D's signed upload flow, confirms the upload, and requests a printing estimate. Supply file_id to reuse an existing upload. Set platform_id, or omit it when the account has exactly one enabled platform.

The price is in USD for the requested quantity and excludes shipping. filament_id selects a public filament ID; omitting it uses Slant3D's default black PLA. The returned file_id can be reused in order items. If Slant3D returns pricing for a different quantity, the block reports an error instead of presenting it as the requested total.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file_url | STL file URL, workspace file, or data URI; ignored when file_id is set | str (file) | No |
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
