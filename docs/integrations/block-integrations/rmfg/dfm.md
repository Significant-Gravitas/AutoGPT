# Rmfg Dfm
<!-- MANUAL: file_description -->
Blocks that check manufacturability (design for manufacturing) of a configured design. A DFM report says whether the configuration is ready, needs input or is blocked, and reports what each part and hole can take.
<!-- END MANUAL -->

## RMFG Create DFM Report

### What it is
Runs an RMFG manufacturability (DFM) check on a configured design. A blocked finding such as a hole close to a bend can be accepted with accepted_risks once the customer agrees; requires_input means a material or profile is still missing

### How it works
<!-- MANUAL: how_it_works -->
Posts the design ID and a manufacturing configuration to `/v1/dfm` with an `Idempotency-Key`. Precedence is per-part override, then `defaults`; the `material_id` input is a shortcut that sets `defaults.material_id`. RMFG answers with a status per part and overall: `ready`, `requires_input` (a selection is missing, listed in `requirements`) or `blocked` (the geometry or configuration cannot be made). Neither of the latter is an error; they are results for the graph to act on. Findings carry a severity, and warning-level ones such as a hole near a bend can be accepted by listing their code in `accepted_risks`. Each part's `capabilities` list which finishes, colors and hardware fit that part and each of its holes.

Unknown design, material or hardware IDs are rejected by RMFG and surfaced as `RMFG <code>: <message> (field: <path>)`. By default production files (laser DXF, corrected STEP) are also prepared; their status arrives on the report and can be re-read with Get DFM Report. Reports are immutable, so a changed configuration means a new report. `review_url` is emitted only when RMFG provides one.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| design_id | Design ID from Analyze Design | str | Yes |
| material_id | Sheet-metal stock for every sheet part, from List Materials. Leave empty for tube-only designs or when configuration sets it. | str | No |
| configuration | Per-part material, tube profile, finish, powder coat, hole operations, welds and accepted risks. A non-empty material_id above overrides defaults.material_id. | ManufacturingConfiguration | No |
| generate_production_files | Also prepare laser DXF and corrected STEP files. | bool | No |
| idempotency_key | Stable key for identical retries; defaults to the node execution ID. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| report | The full DFM report | DFMReport |
| dfm_id | Report ID, for review links and re-reads | str |
| status | ready, requires_input (a selection is missing) or blocked | "requires_input" \| "ready" \| "blocked" |
| is_ready | True when nothing prevents ordering | bool |
| configuration | The configuration that was evaluated; feed it to a quote | ManufacturingConfiguration |
| issues | Every finding across all parts and the assembly | List[DFMIssue] |
| issue | One finding at a time | DFMIssue |
| requirements | Selections still needed before the design can be quoted | List[Requirement] |
| parts | Per-part status, findings, capabilities and images | List[PartDFM] |
| review_url | Website page showing this exact configuration for a person to adjust; emitted when RMFG provides one | str |

### Possible use case
<!-- MANUAL: use_case -->
**Resolving requires_input**: Read the `material_required` requirement and each part's capabilities, pick a compatible material, and re-run.

**Accepting a Known Risk**: After the customer approves a hole-near-bend warning, re-run with its code in `accepted_risks`.

**Production File Prep**: Generate laser DXF and corrected STEP files for a configuration that is already priced.
<!-- END MANUAL -->

---

## RMFG Get DFM Report

### What it is
Fetches an RMFG DFM report by ID

### How it works
<!-- MANUAL: how_it_works -->
Fetches `/v1/dfm/{id}` and emits the same outputs as Create DFM Report. Findings never change, but the `production_files` status does, so re-reading tells you when DXF and STEP files are ready or whether preparation failed with a manual-review warning.

An unknown or foreign report ID is reported as `RMFG not_found_error: <message>`. `issue` is emitted once per finding, so a clean report yields none, and `review_url` only when present.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| dfm_id | Report ID from Create DFM Report | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| report | The full DFM report | DFMReport |
| dfm_id | Report ID, for review links and re-reads | str |
| status | ready, requires_input (a selection is missing) or blocked | "requires_input" \| "ready" \| "blocked" |
| is_ready | True when nothing prevents ordering | bool |
| configuration | The configuration that was evaluated; feed it to a quote | ManufacturingConfiguration |
| issues | Every finding across all parts and the assembly | List[DFMIssue] |
| issue | One finding at a time | DFMIssue |
| requirements | Selections still needed before the design can be quoted | List[Requirement] |
| parts | Per-part status, findings, capabilities and images | List[PartDFM] |
| review_url | Website page showing this exact configuration for a person to adjust; emitted when RMFG provides one | str |

### Possible use case
<!-- MANUAL: use_case -->
**Production Files Ready**: After a `dfm_report.production_files.ready` event, read the report and forward the file links to the shop floor.

**Audit Trail**: Re-read the report a quote was based on to show the customer exactly what was checked.

**Failed Preparation**: Detect a `production_files` failure and route the design to a person for review.
<!-- END MANUAL -->

---
