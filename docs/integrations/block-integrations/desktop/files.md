# Desktop Files
<!-- MANUAL: file_description -->
Read, write and list files on an interactive desktop created by the Desktop Sandbox block, including its persistent workspace volume.
<!-- END MANUAL -->

## Desktop File

### What it is
Reads, writes, or lists files inside an interactive desktop sandbox, including its persistent workspace.

### How it works
<!-- MANUAL: how_it_works -->
Reconnects to the desktop by `sandbox_id` and uses E2B's filesystem API directly, with no shell involved. `read` returns the file's text, `write` creates or overwrites it (creating parent directories as needed), and `list` returns the entry names in a directory. Paths under `/home/user/workspace` live on the mounted volume and survive the sandbox; anything else persists only through suspend and resume.
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
- Drop an input file into `/home/user/workspace` before a desktop task, then read back the result the GUI application saved.
- List the Downloads folder after a browser action to find the file that was just fetched.
<!-- END MANUAL -->

---
