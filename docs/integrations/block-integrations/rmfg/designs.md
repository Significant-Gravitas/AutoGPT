# Rmfg Designs
<!-- MANUAL: file_description -->
Blocks that upload a STEP file to RMFG for analysis and read the resulting design. Analysis is the first step of every quote: it splits an assembly into unique parts and detects bends, holes and the suggested manufacturing process.
<!-- END MANUAL -->

## RMFG Analyze Design

### What it is
Uploads a STEP file to RMFG and returns its analyzed parts

### How it works
<!-- MANUAL: how_it_works -->
Reads the input file (a URL, data URI or workspace file), uploads it as multipart form data to `/v1/analyze`, and by default polls the design until it is ready or has failed. The upload carries an `Idempotency-Key`, defaulting to the node execution ID, so a retried run returns the same design instead of a duplicate. The result lists every unique part with its instance count, dimensions, bends and holes; hole IDs are stable and are what later hole operations refer to. Turn off `wait_for_ready` to get the design ID back at once and fetch the result later with Get Design.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| file | STEP or STP file to analyze (URL, data URI or workspace file) | str (file) | Yes |
| file_name | Name to record for the upload; defaults to the file's own. | str | No |
| wait_for_ready | Poll until analysis finishes instead of returning at once. | bool | No |
| timeout_seconds | How long to wait for analysis when wait_for_ready is on. | int | No |
| idempotency_key | Stable key so a retried upload returns the same design. Defaults to this node execution's ID. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| design | The design resource | Design |
| design_id | ID to pass to DFM, quote and cart | str |
| status | queued, processing, ready or failed | "queued" \| "processing" \| "ready" \| "failed" |
| parts | Every unique part with its instance count, once ready | List[Part] |
| part | One part at a time | Part |
| part_ids | Part IDs in the same order | List[str] |
| review_url | Website page where a person can inspect and configure the design | str |
| image_url | Rendered picture of the whole design | str |

### Possible use case
<!-- MANUAL: use_case -->
A customer emails a STEP file. The agent runs Analyze Design, reads back two unique parts with their instance counts, and shows the customer the rendered `image_url` before asking which material to quote.
<!-- END MANUAL -->

---

## RMFG Get Design

### What it is
Fetches an RMFG design and its analyzed parts by ID

### How it works
<!-- MANUAL: how_it_works -->
Fetches `/v1/designs/{id}`. Optionally polls until analysis has finished, which is useful when Analyze Design was run with `wait_for_ready` off or an earlier run timed out. Outputs are identical to Analyze Design.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| design_id | Design ID from Analyze Design | str | Yes |
| wait_for_ready | Poll until analysis finishes instead of returning at once. | bool | No |
| timeout_seconds | How long to wait for analysis when wait_for_ready is on. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| design | The design resource | Design |
| design_id | ID to pass to DFM, quote and cart | str |
| status | queued, processing, ready or failed | "queued" \| "processing" \| "ready" \| "failed" |
| parts | Every unique part with its instance count, once ready | List[Part] |
| part | One part at a time | Part |
| part_ids | Part IDs in the same order | List[str] |
| review_url | Website page where a person can inspect and configure the design | str |
| image_url | Rendered picture of the whole design | str |

### Possible use case
<!-- MANUAL: use_case -->
A `design.ready` webhook fires; the graph passes the event's `resource_id` into Get Design to load the parts and continue to quoting.
<!-- END MANUAL -->

---
