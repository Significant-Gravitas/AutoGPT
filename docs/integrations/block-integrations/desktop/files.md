# Desktop Files
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Desktop File

### What it is
Reads, writes, or lists files inside an interactive desktop sandbox, including its persistent workspace.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| sandbox_id | ID of the desktop sandbox to access files in | str | Yes |
| operation | File operation to perform | "read" \| "write" \| "list" | No |
| path | File or directory path inside the sandbox; the persistent workspace lives at /home/user/workspace | str | Yes |
| content | Content to write (for the 'write' operation) | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| content | File content (for 'read') | str |
| entries | Directory entries (for 'list') | List[str] |
| path | Path that was operated on | str |
| cost_meter | Estimated provider cost telemetry for this block run | CostMeter |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
