# Rmfg Images
<!-- MANUAL: file_description -->
Block that downloads RMFG's rendered pictures. The API's image links need the account's credentials, so a bare URL is useless downstream; this block fetches the PNG and hands it on as a platform media file.
<!-- END MANUAL -->

## RMFG Get Image

### What it is
Downloads RMFG's rendered picture of a design or part, with holes and bends labelled

### How it works
<!-- MANUAL: how_it_works -->
Requests `/v1/designs/{id}/image`, `/v1/designs/{id}/parts/{part_id}/image`, or `/v1/dfm/{id}/parts/{part_id}/image` with the chosen `view` (iso, top, bottom, front, back, left, right, or flat for a sheet part's flat pattern) and optional `width` (0 uses RMFG's default, up to 4096). Setting `dfm_id` without `part_id` is rejected before any request, since a DFM picture is always of one part. Part pictures label every hole and bend by ID, which is how an agent decides which `hole_id` to tap or countersink; DFM part pictures additionally mark the configured operations.

The block always asks for PNG and checks the bytes it gets back: anything else (the SVG the API can also render, or an HTML error page) is refused with `RMFG returned <type> instead of a PNG` rather than stored as a media file. Unknown IDs and other HTTP errors are reported as `RMFG <code>: <message>`. The PNG goes through the platform's media pipeline, so the output is a workspace file in CoPilot and a data URI in graphs.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| design_id | Design ID from Analyze Design | str | Yes |
| part_id | Draw only this part, with every hole and bend labelled by ID. Empty draws the whole design. | str | No |
| dfm_id | With part_id, draw the part as configured in this DFM report, taps, studs, nuts and countersinks marked. | str | No |
| view | Camera angle; flat shows a sheet part's flat pattern. | "iso" \| "top" \| "bottom" \| "front" \| "back" \| "left" \| "right" \| "flat" | No |
| width | Image width in pixels; 0 uses RMFG's default. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| image | The rendered PNG | str (file) |

### Possible use case
<!-- MANUAL: use_case -->
**Hole Selection**: Show the customer the flat view of a part with labelled holes before asking which to tap.

**Quote Attachment**: Attach an iso render of the whole design to the quote email.

**Configuration Proof**: Send the DFM part picture so the customer can see where the hardware will go.
<!-- END MANUAL -->

---
