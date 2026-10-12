# Rmfg Catalog
<!-- MANUAL: file_description -->
Blocks that read RMFG's catalogs of sheet-metal stock, tube profiles, finishes, powder-coat colors and hardware. Every other RMFG block takes catalog IDs rather than names, so a quoting graph usually starts here. All RMFG blocks accept either an API key from rmfg.com/account or a connected account: choose Connect, open the approval link RMFG shows, and confirm the code.
<!-- END MANUAL -->

## RMFG List Finishes

### What it is
Lists the finishes RMFG can apply to sheet or tube parts

### How it works
<!-- MANUAL: how_it_works -->
Reads `/v1/finishes` in pages of 500, following `next_cursor` until `has_more` is false; the optional `process` filter (`sheet_metal` or `tube_laser`) is sent as a query parameter. The block emits the full list, each finish on `finish` one at a time, and `finish_ids` for wiring into a configuration; an empty catalog gives an empty list and no per-item output.

A non-2xx answer from RMFG is raised as `RMFG <code>: <message>` on the `error` output, with 401/403 pointing at the key and its scopes. A cursor that repeats, or a listing that runs past 100 pages, raises a `pagination_error` instead of looping forever. The list says what exists; whether a finish fits a specific part comes from a DFM report's `capabilities`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| process | Only finishes that apply to this process; empty for all. | "sheet_metal" \| "tube_laser" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| finishes | Matching finishes | List[Finish] |
| finish | One finish at a time | Finish |
| finish_ids | IDs in the same order | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Deburred Bracket Quote**: List sheet-metal finishes, pick the entry named Deburr, and pass its id as `finish_id` in the quote configuration.

**Finish Menu for Customers**: Show a customer the finishes available for their process before they choose one.

**Configuration Validation**: Check that a `finish_id` saved in an old graph still exists before re-quoting with it.
<!-- END MANUAL -->

---

## RMFG List Hardware

### What it is
Lists the taps, studs, nuts or standoffs RMFG can install

### How it works
<!-- MANUAL: how_it_works -->
Reads one of the four `/v1/hardware/{kind}` catalogs (`taps`, `studs`, `nuts` or `standoffs`) with pagination. Each family has its own fields (thread pitch, PEM part number, minimum sheet thickness), which pass through untouched on `option`; the id becomes `tap_id`, `stud_id`, `nut_id` or `standoff_id` on a hole operation in a part configuration.

`kind` is an enum, so an unknown family is rejected before any request is made. API errors and pagination faults are surfaced as on the other catalog blocks, and a family with no entries yields an empty `options` list.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| kind | Which catalog to read. Reference an entry's id as tap_id, stud_id, nut_id or standoff_id in a part configuration. | "taps" \| "studs" \| "nuts" \| "standoffs" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| options | Catalog entries | List[Dict[str, Any]] |
| option | One entry at a time | Dict[str, Any] |
| option_ids | IDs in the same order | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Tapped Holes**: Find the M4 tap entry and use its id in the `taps` array of a part configuration.

**PEM Hardware Lookup**: Look up a self-clinching nut by part number before adding it to a hole.

**Sheet Thickness Check**: Read each stud's minimum sheet thickness and skip options the chosen material is too thin for.
<!-- END MANUAL -->

---

## RMFG List Materials

### What it is
Lists the sheet-metal stock RMFG can cut and bend, with thickness in mm and inches. Choose the entry closest to a part's detected thickness and pass its id as material_id to quote

### How it works
<!-- MANUAL: how_it_works -->
Reads `/v1/materials` across all pages. Each material is a specific alloy at a stock thickness, given in both inches and millimetres, with a `bendable` flag; use the id as `material_id` on quotes, carts and DFM reports. Tube parts use tube profiles instead.

The block takes no inputs beyond credentials, so the only failures are API-side: an invalid or under-scoped key is reported as `RMFG <code>: <message>. Check the RMFG API key and its scopes`, and any other non-2xx answer as `RMFG <code>: <message>`. Rows are validated into a model whose fields all have defaults, so extra fields from a newer API version pass through instead of failing the block.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| materials | Every sheet-metal material, across all pages | List[Material] |
| material | One material at a time | Material |
| material_ids | IDs in the same order | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Closest Stock Thickness**: Pick the material whose `thickness_mm` is nearest a part's `detected_thickness_mm` and report the difference.

**Alloy Request Matching**: Turn "5052 aluminum, about an eighth inch" into the catalog id with `thickness_in` near 0.125.

**Bendable Stock Only**: Filter to `bendable` materials when the part has bends.
<!-- END MANUAL -->

---

## RMFG List Powder Coat Colors

### What it is
Lists the powder-coat colors RMFG offers

### How it works
<!-- MANUAL: how_it_works -->
Reads `/v1/powder-coat-colors` with pagination. Each color has a hex value for previews, an `available` flag and a price multiplier; use the id as `powder_coat_color_id` in a configuration. The DFM report decides whether a given part can be coated.

Colors with `available` false are still listed, so check the flag before offering one. API errors and pagination faults are surfaced as on the other catalog blocks.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| colors | Every color | List[PowderCoatColor] |
| color | One color at a time | PowderCoatColor |
| color_ids | IDs in the same order | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Color by Name**: Match a customer's requested color to a catalog entry and quote the part coated in it.

**Swatch Preview**: Show hex swatches of every available color in a storefront.

**Price Impact**: Compare price multipliers to explain why a metallic color costs more.
<!-- END MANUAL -->

---

## RMFG List Tube Profiles

### What it is
Lists the tube stock profiles RMFG can laser-cut

### How it works
<!-- MANUAL: how_it_works -->
Reads `/v1/tube-profiles` across all pages. A profile is a material plus a cross-section (square, rectangular or round) with outer dimensions and wall thickness in millimetres; use the id as `tube_profile_id` for parts whose `suggested_process` is `tube_laser`.

Round profiles carry `outer_diameter_mm` and leave width and height empty, so read `shape` first. API errors and pagination faults are surfaced as on the other catalog blocks.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| tube_profiles | Every tube profile, across all pages | List[TubeProfile] |
| tube_profile | One profile at a time | TubeProfile |
| tube_profile_ids | IDs in the same order | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Matching Tube Stock**: Choose the profile matching a detected cross-section and configure the tube part with it.

**Wall Thickness Options**: Offer the customer the available wall thicknesses for a 25 mm square tube.

**Stock Length Planning**: Read `default_stock_length_mm` to explain how long a part can be cut.
<!-- END MANUAL -->

---
