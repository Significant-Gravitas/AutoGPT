# Capy Usage
<!-- MANUAL: file_description -->
Track what a Capy organization is spending.
<!-- END MANUAL -->

## Capy Get Usage

### What it is
Reports how much your Capy organization spent in a date range, broken down by member, model and kind of usage.

### How it works
<!-- MANUAL: how_it_works -->
Calls `GET /api/v1/usage`, which defaults to the current month to now. `total_dollars` is the headline figure; `report` carries the breakdown by kind, member, model and image, plus token counts.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| start | Start of the range as an ISO date or timestamp. Empty means the start of the current month. | str | No |
| end | End of the range as an ISO date or timestamp. Empty means now. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| total_dollars | Total spend in the range | float |
| report | The full report: totals by kind (LLM, image, VM), token counts, and breakdowns by member, model and image | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
Send a weekly Capy spend summary, or stop starting new threads once the month's spend passes a budget.
<!-- END MANUAL -->

---
