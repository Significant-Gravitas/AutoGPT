# Specstream Run Status
<!-- MANUAL: file_description -->
Blocks that check on a SpecStream run and wait for it to end.
<!-- END MANUAL -->

## SpecStream Get Run Status

### What it is
Reads the status of a SpecStream dataset run. Costs nothing.

### How it works
<!-- MANUAL: how_it_works -->
The block reads the run once and returns its status, the rows matched so far and what it has charged. It costs nothing and is safe to repeat.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| run_id | The run ID from Approve and Run | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| run_id | The ID of the run | str |
| status | queued, planning, running, completed, partial, failed, canceled or expired | str |
| is_done | True once the run has ended | bool |
| requested_rows | How many rows were asked for | int |
| accepted_rows | How many rows met every condition so far | int |
| charged_cents | What the run has charged so far, in cents | int |
| stop_reason | Why the run ended, empty while it runs | str |

### Possible use case
<!-- MANUAL: use_case -->
**Scheduled Checks**: Check a run from a scheduled workflow before deciding whether to read its rows.

**Progress Reports**: Show how many rows have matched so far to a person watching the workflow.

**Spend Tracking**: Read the amount charged for a finished run and log it next to the rows it paid for.
<!-- END MANUAL -->

---

## SpecStream Wait For Run

### What it is
Waits for a SpecStream dataset run to end by checking its status. Costs nothing.

### How it works
<!-- MANUAL: how_it_works -->
The block checks the run with growing gaps between checks until it ends or the timeout passes. It uses no webhook. A run takes minutes, so the default timeout is 15 minutes. If the timeout passes first, the run keeps going and timed_out is true, so the same run can be read again later.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| run_id | The run ID from Approve and Run | str | Yes |
| timeout | The longest to wait, in seconds. A run takes minutes, and a large one more. If this passes, the run keeps going and can be read again. | int | No |
| check_interval | Seconds between the first status checks | int | No |
| max_interval | The longest gap between checks, as the gap grows | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| run_id | The ID of the run | str |
| status | The last status read | str |
| is_done | True if the run ended in time | bool |
| timed_out | True if the timeout passed first | bool |
| elapsed_seconds | How long the block waited | float |
| accepted_rows | How many rows have matched | int |
| charged_cents | What the run charged, in cents | int |

### Possible use case
<!-- MANUAL: use_case -->
**Run Sequencing**: Place this block between Approve And Run and List Rows so the rows are read after the run has ended.

**Timeout Branching**: Send the workflow down a different path when the wait ends before the run does, and read the run again later.

**Report on Completion**: Hold a report or a message until the dataset is complete, with no webhook to set up.
<!-- END MANUAL -->

---
