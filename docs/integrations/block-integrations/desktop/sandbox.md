# Desktop Sandbox
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Create Desktop Sandbox

### What it is
Creates (or reconnects to) an interactive cloud desktop with a live, embeddable stream and a persistent mounted workspace.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| workspace_scope | Which persistent workspace volume to mount: 'user' gives every sandbox you create the same durable workspace; 'agent' gives this agent its own durable workspace. | "user" \| "agent" | No |
| sandbox_id | Reconnect to an existing desktop sandbox instead of creating a new one. Suspended sandboxes resume automatically. | str | No |
| width | Screen width in pixels | int | No |
| height | Screen height in pixels | int | No |
| timeout_minutes | Idle timeout in minutes after which the desktop auto-suspends (state is preserved and resumes on reconnect). | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| sandbox_id | ID of the desktop sandbox | str |
| desktop_stream | Live interactive desktop stream; the URL is directly embeddable and viewable in the browser. | DesktopStream |
| workspace_path | Path of the persistent workspace directory inside the sandbox | str |
| persistence | Whether a persistent volume is mounted for the workspace | PersistenceInfo |
| cost_meter | Estimated provider cost telemetry for this block run | CostMeter |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Stop Desktop Sandbox

### What it is
Suspends (default) or destroys an interactive desktop sandbox. Suspended desktops preserve all state and can be resumed.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| sandbox_id | ID of the desktop sandbox to stop | str | Yes |
| mode | 'suspend' preserves the full desktop state for later resume; 'destroy' permanently deletes the sandbox (mounted workspace volumes survive). | "suspend" \| "destroy" | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| sandbox_id | ID of the stopped sandbox | str |
| final_status | Resulting sandbox state: 'suspended' or 'destroyed' | str |
| cost_meter | Estimated provider cost telemetry for this block run | CostMeter |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
