# Exa Search Monitors Batch
<!-- MANUAL: file_description -->
Blocks for deleting, pausing or unpausing many Exa Search Monitors at once by filter.
<!-- END MANUAL -->

## Exa Batch Search Monitors

### What it is
Delete, pause or unpause every Exa Search Monitor matching a name, status or metadata filter (previews by default)

### How it works
<!-- MANUAL: how_it_works -->
This block sends `POST /monitors/batch` to Exa with the action, a filter built from name, status and metadata, `dry_run` and `limit`. At least one filter must be set; the block refuses to act on every monitor at once. `dry_run` is on by default, so the first run only reports which monitors would be affected; turn it off to apply the action. Each call handles up to `limit` monitors (1-500); run it again while `has_more` is true. The block does not charge for the call. The block is flagged as an irreversible action because `delete` cannot be undone. It needs your own Exa API key and fails with a clear message on the platform's shared Exa key, since every user of that key shares one Exa account.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| action | delete removes matching monitors permanently; pause stops their scheduled runs; unpause makes them active again | "delete" \| "pause" \| "unpause" | Yes |
| name | Match monitors whose name contains this text (case-insensitive) | str | No |
| status | Match monitors in this status | "active" \| "paused" \| "disabled" | No |
| metadata | Match monitors whose metadata has all of these pairs | Dict[str, str] | No |
| dry_run | Only preview which monitors would be affected. Turn off to actually apply the action. | bool | No |
| limit | Most monitors to process in one call (1-500). Run again while has_more is true to process the rest. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| action | The action requested | str |
| affected | How many monitors were (or, in a dry run, would be) affected | int |
| monitor_ids | IDs of those monitors | List[str] |
| dry_run | Whether this was only a preview | bool |
| has_more | More monitors match; run again to process them | bool |

### Possible use case
<!-- MANUAL: use_case -->
**Client Offboarding**: Delete every monitor whose metadata carries a departing client's ID, after checking the dry-run list.

**Pausing a Project**: Pause all monitors whose name contains a project name while that project is on hold.

**Resuming Work**: Unpause every paused monitor with a given metadata tag when the work starts again.
<!-- END MANUAL -->

---
