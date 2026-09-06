# Rmfg Catalog
<!-- MANUAL: file_description -->
Blocks that read RMFG's catalogs of sheet-metal stock, tube profiles, finishes, powder-coat colors and hardware. Every other RMFG block takes catalog IDs rather than names, so a quoting graph usually starts here.
<!-- END MANUAL -->

## RMFG List Finishes

### What it is
Lists the finishes RMFG can apply to sheet or tube parts

### How it works
<!-- MANUAL: how_it_works -->
Reads `/v1/finishes`, following pagination until every entry is returned. Filter by `process` to see only finishes that apply to sheet-metal or tube-laser parts. The list says what exists; whether a finish fits a specific part comes from a DFM report's capabilities. The block emits the full list, each finish individually, and the bare IDs for wiring into a configuration.
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
An agent quoting a bracket wants it deburred: it lists finishes for sheet metal, picks the entry named Deburr, and passes its id as `finish_id` in the quote configuration.
<!-- END MANUAL -->

---

## RMFG List Hardware

### What it is
Lists the taps, studs, nuts or standoffs RMFG can install

### How it works
<!-- MANUAL: how_it_works -->
Reads one of the four `/v1/hardware/*` catalogs — taps, studs, nuts or standoffs — with pagination. Each family has its own fields (thread pitch, PEM part number, minimum sheet thickness), which pass through untouched. The id becomes `tap_id`, `stud_id`, `nut_id` or `standoff_id` on a hole operation in a part configuration.
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
Before adding M4 taps to two holes, the agent lists taps, finds the M4 entry, and uses its id in the `taps` array of the part's configuration; a DFM report then confirms the holes are the right diameter.
<!-- END MANUAL -->

---

## RMFG List Materials

### What it is
Lists the sheet-metal materials RMFG can cut and bend

### How it works
<!-- MANUAL: how_it_works -->
Reads `/v1/materials` across all pages. Each material is a specific alloy at a specific stock thickness, given in both inches and millimetres, with a `bendable` flag. Use the id as `material_id` on quotes, carts and DFM reports. Tube parts use tube profiles instead.
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
A user asks for "the bracket in 5052 aluminum, about an eighth inch". The agent lists materials, filters to 5052 with `thickness_in` near 0.125, and quotes with that id.
<!-- END MANUAL -->

---

## RMFG List Powder Coat Colors

### What it is
Lists the powder-coat colors RMFG offers

### How it works
<!-- MANUAL: how_it_works -->
Reads `/v1/powder-coat-colors` with pagination. Each color has a hex value for previews, an `available` flag and a price multiplier. Use the id as `powder_coat_color_id` in a configuration; the DFM report decides whether a given part can be coated.
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
A storefront agent lets a customer pick a color by name, matches it to the catalog entry, and quotes the part powder-coated in that color.
<!-- END MANUAL -->

---

## RMFG List Tube Profiles

### What it is
Lists the tube stock profiles RMFG can laser-cut

### How it works
<!-- MANUAL: how_it_works -->
Reads `/v1/tube-profiles` across all pages. A profile is a material plus a cross-section — square, rectangular or round — with outer dimensions and wall thickness in millimetres. Use the id as `tube_profile_id` for parts whose `suggested_process` is `tube_laser`.
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
After analysis reports a tube part, the agent lists profiles, chooses the one matching the detected cross-section, and configures the part with it.
<!-- END MANUAL -->

---
