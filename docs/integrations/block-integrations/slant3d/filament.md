# Slant3D Filament
<!-- MANUAL: file_description -->
Blocks for getting available filament options from Slant3D.
<!-- END MANUAL -->

## Slant3D Filament

### What it is
Get list of available filaments

### How it works
<!-- MANUAL: how_it_works -->
This block queries the Slant3D API to retrieve a list of all available filament options for 3D printing. Each filament includes details like color, material type, and availability.

Use this to populate filament selection dropdowns or validate filament choices before placing orders.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| filaments | List of available filaments | List[Filament] |

### Possible use case
<!-- MANUAL: use_case -->
**Product Configurator**: Display available filament options in a custom 3D printing order form.

**Inventory Display**: Show current filament availability to customers before they place orders.

**Material Validation**: Verify that requested filaments are available before processing orders.
<!-- END MANUAL -->

---
