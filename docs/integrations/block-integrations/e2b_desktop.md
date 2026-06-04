# E2B Desktop Sandbox Blocks

## What it is
The E2B Desktop Sandbox blocks give AI agents a secure, isolated virtual **graphical** Linux desktop (Ubuntu + Xfce). Built on [E2B's Desktop Sandbox](https://e2b.dev), they let an agent spin up a full desktop, watch it over a live stream, drive the mouse and keyboard, and capture screenshots — the "computer use" workflow.

For running code or shell commands in a **headless** sandbox (no GUI), use the Code Executor blocks (`Execute Code`, `Instantiate Code Sandbox`, `Execute Code Step`) instead. The Desktop blocks use their own **E2B Desktop** credential (separate from the Code Executor's E2B credential), so desktop compute is billed to a dedicated key — but you can paste the same key from your E2B account into both.

## What it does
These blocks enable agents to:
- Spin up a virtual desktop with a live, browser-embeddable stream URL
- List the desktop sandboxes on your account to reconnect or audit them
- Control the mouse and keyboard (click, move, scroll, type, press keys)
- Take screenshots of the current desktop state
- Pause a sandbox to stop compute billing while keeping its full state
- Clean up and kill sandboxes to stop billing immediately

## How it works
Each sandbox is a fully isolated microVM running Linux + Xfce. The blocks use the `e2b-desktop` SDK to communicate with the sandbox over E2B's API. The SDK is synchronous, so each call runs in a worker thread to keep the executor responsive.

The typical "computer use" loop:
1. **Create** a desktop sandbox and start the live stream
2. **Screenshot** the desktop — the "see" step
3. **Control** — perform a click, type text, or press a key based on what was seen
4. Repeat 2–3 until the task is done
5. **Kill** the sandbox to stop billing

## Prerequisites
- An [E2B account](https://e2b.dev) and API key, added as an **E2B Desktop** credential (you can reuse the same key as the Code Executor blocks)
- E2B Pro plan recommended for sessions longer than 1 hour and custom CPU/RAM

## Blocks

### E2B Desktop Create Block

#### What it does
Creates a new E2B Desktop sandbox, optionally runs setup commands, starts a live stream, and returns the sandbox ID and stream URL.

#### Inputs

| Input | Description |
|-------|-------------|
| Credentials | E2B API key. Get one at [e2b.dev](https://e2b.dev/docs) |
| Template ID | Optional E2B desktop template ID for pre-baked environments (skips setup time) |
| Setup Commands | Shell commands to run after sandbox creation (e.g. `git clone`, `npm install`) |
| Timeout | Sandbox lifetime in seconds (default: 3600 = 1 hour; max 86400 on Pro) |
| Stream Require Auth | Whether to password-protect the stream URL (default: true — always recommended) |
| Width | Desktop screen width in pixels (default: 1280). Lower resolutions stream more smoothly |
| Height | Desktop screen height in pixels (default: 720) |
| DPI | Desktop DPI (default: 96). Raise to scale up UI on high-resolution screens |
| Smooth Stream | Re-tune the VNC server for high-motion content (default: true). The stock stream refreshes only ~20 FPS, making animations look laggy; this raises the poll/update rate for a much smoother stream at the cost of more sandbox CPU and bandwidth |

#### Outputs

| Output | Description |
|--------|-------------|
| stream\_url | Live stream URL — rendered inline as an interactive iframe with a full-screen button (press Esc to exit); embed it to watch and control the desktop (mouse + keyboard) in real time |
| auth\_key | Authentication key required to view the stream (already included in the stream\_url) |
| sandbox\_id | Unique ID of the running sandbox — pass to all other blocks |
| error | Error message if sandbox creation failed |

---

### E2B Desktop List Block

#### What it does
Lists the desktop sandboxes tied to your API key — running, paused, or both. Use it to reconnect to an existing desktop (pass a returned `sandbox_id` to any other block) instead of creating a new one, or to audit what is still alive and billing.

#### Inputs

| Input | Description |
|-------|-------------|
| Credentials | E2B API key — lists the sandboxes owned by this key |
| State | Which sandboxes to list: `all`, `running`, or `paused` (default: `all`) |
| Limit | Maximum number of sandboxes to return (default: 100) |

#### Outputs

| Output | Description |
|--------|-------------|
| sandboxes | All matching sandboxes, each with its ID, template ID, state, start/end time, and metadata |
| sandbox | Each matching sandbox, yielded one at a time for downstream iteration |
| count | Number of sandboxes returned |
| error | Error message if the listing failed |

---

### E2B Desktop Control Block

#### What it does
Drives the mouse and keyboard of a running desktop sandbox. This is the "act" half of a computer-use loop — pair it with the Screenshot block to see, then act, then see again.

#### Inputs

| Input | Description |
|-------|-------------|
| Credentials | E2B API key |
| Sandbox ID | ID from `E2B Desktop Create Block` |
| Action | Which input action to perform: `left_click`, `double_click`, `right_click`, `middle_click`, `move_mouse`, `scroll`, `type`, or `press` |
| X / Y | Coordinates for click/move actions. Leave empty for clicks to use the current cursor position; required for `move_mouse` |
| Text | Text to type (used by the `type` action) |
| Keys | Key or combo to press (used by the `press` action), e.g. `enter`, `backspace`, or `ctrl+c` |
| Scroll Direction | `up` or `down` (used by the `scroll` action) |
| Scroll Amount | Number of scroll steps (used by the `scroll` action) |

#### Outputs

| Output | Description |
|--------|-------------|
| success | True if the action was performed |
| error | Error message if the action failed |

---

### E2B Desktop Screenshot Block

#### What it does
Takes a screenshot of the current desktop state and stores it in the AutoGPT workspace, ready to feed into a vision model, post to a PR comment, or use for visual QA.

#### Inputs

| Input | Description |
|-------|-------------|
| Credentials | E2B API key |
| Sandbox ID | ID from `E2B Desktop Create Block` |

#### Outputs

| Output | Description |
|--------|-------------|
| image | The captured screenshot (a workspace reference in CoPilot, a data URI in graphs) — feed directly into downstream blocks |
| error | Error message if screenshot failed |

---

### E2B Desktop Pause Block

#### What it does
Pauses a running sandbox, preserving its filesystem and memory so it can be resumed later. Billing stops entirely while paused — E2B does not charge for paused sandboxes, they're kept indefinitely, and they don't count toward your concurrency limit. Resuming is automatic — pass the returned `sandbox_id` to any other E2B Desktop block and the sandbox wakes up where it left off. Use this instead of Kill whenever you might want the sandbox back.

> Note: the live stream drops while a sandbox is paused — restart it after the sandbox resumes.

#### Inputs

| Input | Description |
|-------|-------------|
| Credentials | E2B API key |
| Sandbox ID | ID from `E2B Desktop Create Block` to pause |

#### Outputs

| Output | Description |
|--------|-------------|
| sandbox\_id | ID of the paused sandbox — pass to any block later to resume it |
| error | Error message if the pause failed |

---

### E2B Desktop Kill Block

#### What it does
Destroys a sandbox immediately. Billing stops within seconds. Always call this when done to avoid runaway costs.

#### Inputs

| Input | Description |
|-------|-------------|
| Credentials | E2B API key |
| Sandbox ID | ID of sandbox to kill |

#### Outputs

| Output | Description |
|--------|-------------|
| success | True if the sandbox was destroyed successfully |
| error | Error message if kill failed |

---

## Pricing
E2B Desktop sandboxes are billed per second while running:
- ~$0.10/hour for 2 vCPU (default)
- ~$0.016/GiB/hour for RAM
- **Sleeping/killed sandboxes cost $0**

A typical 2-hour session costs ~$0.26. Always call `E2B Desktop Kill Block` when done.

See [E2B Pricing](https://e2b.dev/pricing) for full details.

## Example Use Case: Computer-Use Loop

An agent driving a desktop app can:
1. **Create** a desktop sandbox and start the live stream
2. **Screenshot** the desktop and send it to a vision model
3. **Control** — click the button or type into the field the model identified
4. **Screenshot** again to confirm the result, repeating until done
5. **Kill** the sandbox when the task is complete

The live `stream_url` lets a human watch the whole session in their browser.
