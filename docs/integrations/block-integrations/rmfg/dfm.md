# Rmfg Dfm
<!-- MANUAL: file_description -->
Blocks that check manufacturability (design for manufacturing) of a configured design. A DFM report says whether the configuration is ready, needs input or is blocked, and reports what each part and hole can take.
<!-- END MANUAL -->

## RMFG Create DFM Report

### What it is
Runs an RMFG manufacturability (DFM) check on a configured design

### How it works
<!-- MANUAL: how_it_works -->
Posts the design ID and a manufacturing configuration to `/v1/dfm`. Configuration precedence is per-part override, then defaults; the `material_id` input is a shortcut that sets `defaults.material_id`. RMFG responds with a status per part and overall: `ready`, `requires_input` (a selection is missing — see `requirements`) or `blocked` (the geometry or configuration cannot be made). Findings carry a severity, and warning-level ones such as a hole near a bend can be accepted by listing the code in `accepted_risks`. The `capabilities` on each part list which finishes, colors and hardware fit that part and each hole, which is how an agent chooses valid options. By default RMFG also prepares production files (laser DXF, corrected STEP). Reports are immutable: to evaluate a change, create a new one.
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
| review_url | Website page showing this exact configuration for a person to adjust | str |

### Possible use case
<!-- MANUAL: use_case -->
An agent quotes a bracket and gets `requires_input`. It creates a DFM report, reads the `material_required` requirement and the part's capabilities, picks a compatible material, and re-quotes.
<!-- END MANUAL -->

---

## RMFG Get DFM Report

### What it is
Fetches an RMFG DFM report by ID

### How it works
<!-- MANUAL: how_it_works -->
Fetches `/v1/dfm/{id}`. Findings never change, but the `production_files` status does, so re-reading tells you when DXF and STEP files are ready or whether preparation failed with a manual-review warning.
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
| review_url | Website page showing this exact configuration for a person to adjust | str |

### Possible use case
<!-- MANUAL: use_case -->
After a `dfm_report.production_files.ready` webhook, the graph reads the report and forwards the production file links to the shop floor.
<!-- END MANUAL -->

---
