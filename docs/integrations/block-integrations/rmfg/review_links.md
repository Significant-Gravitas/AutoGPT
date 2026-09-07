# Rmfg Review Links
<!-- MANUAL: file_description -->
Blocks for handing a design to a person on rmfg.com. A review link opens the 3D configurator with a starting configuration; whatever the person saves can be read back and used for quoting.
<!-- END MANUAL -->

## RMFG Create Review Link

### What it is
Creates an RMFG review link so a person can inspect and adjust a design

### How it works
<!-- MANUAL: how_it_works -->
Posts a design ID and configuration to `/v1/review-links`. Optionally attach a DFM report so the page opens with that report's exact configuration. The returned `review_url` is a private hand-off link: the viewer can change material, finish and hole operations and save. Every design also carries a default `review_url`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| design_id | Design ID from Analyze Design | str | Yes |
| configuration | Starting configuration shown on the page; may be empty. | ManufacturingConfiguration | No |
| dfm_id | Attach a DFM report so the page opens that report's exact configuration. | str | No |
| idempotency_key | Stable key for identical retries; defaults to the node execution ID. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| review_link | The review link resource | ReviewLink |
| link_id | Link ID, to read the saved result back | str |
| review_url | Website page for a person to inspect and configure the design; keep private | str |
| configuration | The configuration on the link, including any saved changes | ManufacturingConfiguration |
| configuration_updated_at | When a person last saved changes; empty until they do | str |
| status | open or expired | str |

### Possible use case
<!-- MANUAL: use_case -->
A DFM report comes back `requires_input` on an assembly with many holes. Rather than guess, the agent creates a review link and asks the customer to configure the parts themselves.
<!-- END MANUAL -->

---

## RMFG Get Review Link

### What it is
Fetches an RMFG review link and the configuration a person saved on it

### How it works
<!-- MANUAL: how_it_works -->
Fetches `/v1/review-links/{id}`. `configuration_updated_at` is set once a person has saved changes; `configuration` then holds their choices, ready to pass into Create Quote or Create Cart.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| link_id | Link ID from Create Review Link, or a design's review_link_id | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| review_link | The review link resource | ReviewLink |
| link_id | Link ID, to read the saved result back | str |
| review_url | Website page for a person to inspect and configure the design; keep private | str |
| configuration | The configuration on the link, including any saved changes | ManufacturingConfiguration |
| configuration_updated_at | When a person last saved changes; empty until they do | str |
| status | open or expired | str |

### Possible use case
<!-- MANUAL: use_case -->
After the customer replies "done", the agent reads the link, sees `configuration_updated_at` is set, and quotes the saved configuration.
<!-- END MANUAL -->

---
