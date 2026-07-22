# Tenki Code Execution
<!-- MANUAL: file_description -->
Run commands in isolated, ephemeral Tenki cloud sandboxes from an AutoGPT workflow. Connect a Tenki API key in AutoGPT before using this block.
<!-- END MANUAL -->

## Tenki Run Code

### What it is
Run a shell command in a fresh Tenki cloud sandbox. The sandbox is always terminated after the command finishes or fails.

### How it works
<!-- MANUAL: how_it_works -->
The block creates a fresh sandbox with inbound access disabled, waits for it to become ready, and runs the command in `/home/tenki` unless another working directory is provided. If the API key has access to one project, the block selects it automatically; otherwise, provide the project ID.

Each sandbox has a hard lifetime limit equal to the startup timeout, command timeout, and a 60-second cleanup margin. The block also terminates the sandbox after success, command failure, timeout, or workflow cancellation, then returns command output and timing details. See the [Tenki documentation](https://www.tenki.cloud/docs) for API key and sandbox concepts.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| command | Shell command to run in a fresh Tenki sandbox | str | Yes |
| project_id | Tenki project ID. Leave empty when the API key has access to exactly one project. | str | No |
| working_directory | Sandbox working directory; empty uses /home/tenki | str | No |
| environment | Environment variables passed only to the command | Dict[str, str] | No |
| timeout_seconds | Maximum command runtime in seconds | int | No |
| startup_timeout_seconds | Maximum time to wait for the sandbox to become ready | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| stdout | Command standard output | str |
| stderr | Command standard error | str |
| exit_code | Command exit code | int |
| duration_seconds | Command execution time in seconds | float |
| startup_time_seconds | Sandbox create-to-ready time in seconds | float |
| sandbox_id | ID of the terminated ephemeral sandbox | str |

### Possible use case
<!-- MANUAL: use_case -->
**Isolated Script Execution**: Run generated or user-provided scripts without exposing the AutoGPT backend host.

**Dependency Validation**: Install a package and execute a smoke test in a disposable Linux environment.

**Build and Test Automation**: Clone a repository, run its checks, and return the logs to later workflow blocks.
<!-- END MANUAL -->

---
