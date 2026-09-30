# Rmfg Designs
<!-- MANUAL: file_description -->
Blocks that upload a STEP file to RMFG for analysis and read the resulting design. Analysis is the first step of every quote: it splits an assembly into unique parts and detects bends, holes and the suggested manufacturing process.
<!-- END MANUAL -->

## RMFG Analyze Design

### What it is
Uploads a STEP file to RMFG, a manufacturer that makes and ships real sheet-metal and tube-laser parts, and returns its analyzed parts: the first step before a live quote, cart or order. Works on a URL, an attached workspace file or a data URI

### How it works
<!-- MANUAL: how_it_works -->
Reads the input file through the platform's media pipeline (a URL, data URI or workspace file), appends `.step` to the recorded name unless it already ends in `.step` or `.stp`, and uploads it as multipart form data to `/v1/analyze` with an `Idempotency-Key` (defaulting to the node execution ID), so a retried run returns the same design instead of a duplicate. The request asks RMFG to hold the connection for up to 20 seconds; with `wait_for_ready` on, the block then polls with growing delays, capped by `timeout_seconds`, until the design is `ready` or `failed`.

A `failed` analysis (for example a file that is not a solid body) raises `RMFG design_failed: Analysis failed: <reason>`; running out of time raises `Timed out after Ns waiting for design <id>; fetch it again later with its ID`, and any HTTP error is reported as `RMFG <code>: <message>`. `review_url` and `image_url` are only emitted once RMFG has assigned them, and `part` is emitted once per unique part, so a design that is still queued has no parts yet.
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
| review_url | Website page where a person can inspect and configure the design; emitted once RMFG has assigned one | str |
| image_url | Rendered picture of the whole design; emitted once analysis is ready | str |

### Possible use case
<!-- MANUAL: use_case -->
**Customer STEP Upload**: Analyze an emailed STEP file and show the customer each unique part and its instance count before quoting.

**Assembly Breakdown**: Split an assembly into unique parts so every one can be configured and priced.

**Fire and Forget**: Upload with `wait_for_ready` off and let a `design.ready` webhook continue the graph.
<!-- END MANUAL -->

---

## RMFG Get Design

### What it is
Fetches an RMFG design and its analyzed parts by ID

### How it works
<!-- MANUAL: how_it_works -->
Fetches `/v1/designs/{id}` and emits the same outputs as Analyze Design. With `wait_for_ready` on it polls until analysis leaves `queued`/`processing`, bounded by `timeout_seconds`, which is useful after an upload with waiting off or a run that timed out.

An unknown ID is reported as `RMFG not_found_error: <message>`. A design whose analysis failed raises `RMFG design_failed` with the reason when waiting, or is returned with `status` `failed` when not. `review_url` and `image_url` are emitted only when present.
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
| review_url | Website page where a person can inspect and configure the design; emitted once RMFG has assigned one | str |
| image_url | Rendered picture of the whole design; emitted once analysis is ready | str |

### Possible use case
<!-- MANUAL: use_case -->
**Webhook Continuation**: Load the parts after a `design.ready` event using the event's `resource_id`.

**Resume After Timeout**: Re-read a large design that Analyze Design gave up waiting for.

**Part ID Lookup**: Re-fetch a design's `part_ids` before building a per-part configuration.
<!-- END MANUAL -->

---
