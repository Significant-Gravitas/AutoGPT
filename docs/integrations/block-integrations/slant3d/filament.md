# Slant3D Filament
<!-- MANUAL: file_description -->
Blocks for getting available filament options from Slant3D.
<!-- END MANUAL -->

## Slant3D Filament

### What it is
Get available filaments, their public IDs, and material and color details

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Slant3D API to retrieve a list of all available filament options for 3D printing. Each filament includes details like color, material type, and availability.

Use this to populate filament selection dropdowns or validate filament choices before placing orders.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| profiles | Filter materials; empty returns all | List["PLA" \| "PETG" \| "OPM"] | No |
| colors | Filter color names; empty returns all | List[str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| filaments | Available filaments; use publicId as filament_id | List[Filament] |

### Possible use case
<!-- MANUAL: use_case -->
**Product Configurator**: Display available filament options in a custom 3D printing order form.

**Inventory Display**: Show current filament availability to customers before they place orders.

**Material Validation**: Verify that requested filaments are available before processing orders.
<!-- END MANUAL -->

---
