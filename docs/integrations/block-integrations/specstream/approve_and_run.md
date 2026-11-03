# Specstream Approve And Run
<!-- MANUAL: file_description -->
Blocks that approve a SpecStream quote and start the run.
<!-- END MANUAL -->

## SpecStream Approve And Run

### What it is
Approves a SpecStream quote at the stated maximum and starts the dataset run. The run takes minutes and is charged per matching row.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the quote ID, its signature and the maximum in cents that you approve. SpecStream starts the run only when that maximum equals the quote, holds it from your balance, and charges for matching rows only. Anything the run does not use goes back to the balance. The block is marked as an irreversible action so the platform can pause for a person to approve it. A retry with the same key never starts a second run, and the default key is made from the quote ID.

Turn on safe mode for sensitive actions in the settings of the workflow that uses this block. It is off by default, and with it off nobody approves the run. The quote also expires 30 minutes after it is created, so a slow approval ends in an expired quote error. Create a new quote then.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| quote_id | The quote ID from Create Dataset Request | str | Yes |
| quote_hash | The quote signature from Create Dataset Request | str | Yes |
| confirm_max_cents | The most you approve this run to cost, in cents. It must equal the maximum_cents of the quote, or nothing runs. Only matching rows are charged, and what the run does not use goes back to your balance. | int | Yes |
| idempotency_key | A stable key so a retry never starts a second run. Defaults to one made from the quote ID, and a quote starts one run at most. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| run_id | The ID of the run, for Get Run Status and List Rows | str |
| state | The state of the approved work: reserved, running, committed or released | str |
| reserved_cents | The maximum held from your balance for this run, in cents | int |
| replay | True when this key had already approved this quote | bool |

### Possible use case
<!-- MANUAL: use_case -->
Put this block after Create Dataset Request, with safe mode on so a person approves the amount, then wait for the run and read its rows.
<!-- END MANUAL -->

---
