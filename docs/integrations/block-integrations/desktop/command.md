# Desktop Command
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Desktop Command

### What it is
Runs a shell command inside an interactive desktop sandbox (DISPLAY is set, so GUI apps can be launched).

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
