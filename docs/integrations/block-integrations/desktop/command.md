# Desktop Command
<!-- MANUAL: file_description -->
Shell access to an interactive desktop created by the Desktop Sandbox block, with the X display wired up so commands can launch and control GUI applications.
<!-- END MANUAL -->

## Desktop Command

### What it is
Runs a shell command inside an interactive desktop sandbox (DISPLAY is set, so GUI apps can be launched).

### How it works
<!-- MANUAL: how_it_works -->
Reconnects to the desktop by `sandbox_id` and runs the command through E2B's command API with `DISPLAY=:0` set, so `firefox https://example.com` or `libreoffice report.odt` opens on the visible desktop. Output is captured until the process exits or `timeout_seconds` elapses; long-lived programs should be backgrounded with `&`. Commands run as the sandbox user with the persistent workspace available at `/home/user/workspace`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| sandbox_id | ID of the desktop sandbox to run the command in | str | Yes |
| command | Shell command to execute inside the desktop sandbox | str | Yes |
| cwd | Working directory for the command | str | No |
| timeout_seconds | Command timeout in seconds | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| stdout | Standard output of the command | str |
| stderr | Standard error of the command | str |
| exit_code | Exit code of the command | int |
| cost_meter | Estimated provider cost telemetry for this block run | CostMeter |

### Possible use case
<!-- MANUAL: use_case -->
- Launch a browser or office application on the desktop before handing control to the Desktop Action block.
- Install a tool with `apt` or `pip`, convert files, or inspect the machine without leaving the workflow.
<!-- END MANUAL -->

---
