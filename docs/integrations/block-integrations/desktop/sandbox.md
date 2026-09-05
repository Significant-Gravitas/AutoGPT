# Desktop Sandbox
<!-- MANUAL: file_description -->
Blocks for running an interactive cloud desktop on E2B: create or reconnect to a desktop with a live, embeddable stream and a durable mounted workspace, then suspend or destroy it when you are done. Pair them with the Desktop Action, Desktop Command and Desktop File blocks to drive the machine. Every desktop block bills per second of running time through your E2B credentials and reports an estimated provider cost in `cost_meter`.
<!-- END MANUAL -->

## Create Desktop Sandbox

### What it is
Creates (or reconnects to) an interactive cloud desktop with a live, embeddable stream and a persistent mounted workspace.

### How it works
<!-- MANUAL: how_it_works -->
Creates an E2B sandbox from the `desktop` template (XFCE on Xvfb) or reconnects to `sandbox_id`, resuming it if it was suspended. A durable volume is mounted at `/home/user/workspace`: the user's shared volume with `workspace_scope: user`, or this agent's own volume with `agent`. The desktop's Downloads, Desktop and Documents folders are redirected into that volume, so files created through the GUI survive the sandbox. The block then starts x11vnc and a noVNC proxy and returns a password-protected stream URL that the frontend renders as an interactive iframe. The sandbox suspends itself after `timeout_minutes` (suspended time is free) and resumes on the next block that uses it. If volumes are unavailable the desktop still works with suspend/resume persistence only, and `persistence.warning` says so.
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
- Let an agent open a real browser to complete a signup, download a report, or work inside a web app while you watch and take over in the live stream.
- Give a long-running agent a persistent machine: create it with `workspace_scope: agent`, save results to `/home/user/workspace`, and reconnect to the same `sandbox_id` on the next run.
<!-- END MANUAL -->

---

## Stop Desktop Sandbox

### What it is
Suspends (default) or destroys an interactive desktop sandbox. Suspended desktops preserve all state and can be resumed.

### How it works
<!-- MANUAL: how_it_works -->
`suspend` pauses the sandbox: E2B snapshots memory and disk, billing stops, and the same `sandbox_id` resumes in about a second with every window where it was. `destroy` kills the sandbox for good; anything on the mounted workspace volume remains, everything else on the machine is gone.
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
- Suspend at the end of a workflow so the next run resumes exactly where it left off without paying for idle time.
- Destroy once a one-off task is finished, keeping only the files that were written to the workspace.
<!-- END MANUAL -->

---
