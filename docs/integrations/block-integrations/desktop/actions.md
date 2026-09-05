# Desktop Actions
<!-- MANUAL: file_description -->
Computer-use input for an interactive desktop created by the Desktop Sandbox block: mouse, keyboard, scrolling, waiting and screenshots, driven by pixel coordinates on the live screen.
<!-- END MANUAL -->

## Desktop Action

### What it is
Performs a computer-use action (mouse, keyboard, scroll, screenshot) on an interactive desktop sandbox.

### How it works
<!-- MANUAL: how_it_works -->
Reconnects to the desktop by `sandbox_id` (resuming it if suspended) and translates the action into `xdotool` commands on the sandbox's X display. Clicks and drags move the pointer to `(x, y)` first; `type` sends text in short chunks with a small key delay so applications keep up; `press` maps friendly key names such as `enter`, `ctrl` or `cmd` to X keysyms and sends them as one chord. `screenshot`, and any action with `screenshot_after` set, captures the screen with `scrot` and returns it as a PNG file so a vision model can decide the next step. Coordinates are screen pixels at the resolution the desktop was created with.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| sandbox_id | ID of the desktop sandbox to act on | str | Yes |
| action | The computer-use action to perform | "screenshot" \| "left_click" \| "double_click" \| "right_click" \| "middle_click" \| "move" \| "drag" \| "scroll" \| "type" \| "press" \| "wait" | No |
| x | X coordinate for click/move/drag-start actions | int | No |
| y | Y coordinate for click/move/drag-start actions | int | No |
| to_x | Destination X coordinate for drag | int | No |
| to_y | Destination Y coordinate for drag | int | No |
| text | Text to type (for the 'type' action) | str | No |
| keys | Keys to press together for the 'press' action, e.g. ['enter'] or ['ctrl', 'c'] | List[str] | No |
| scroll_direction | Scroll direction (for the 'scroll' action) | "up" \| "down" | No |
| scroll_amount | Scroll clicks (for the 'scroll' action) | int | No |
| seconds | Seconds to wait (for the 'wait' action) | float | No |
| screenshot_after | Capture a screenshot of the desktop after the action | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| result | Description of the performed action | str |
| screenshot | Screenshot of the desktop after the action | str (file) |
| cost_meter | Estimated provider cost telemetry for this block run | CostMeter |

### Possible use case
<!-- MANUAL: use_case -->
- Drive a GUI-only application in a loop: screenshot, let a vision model pick the next click or keystroke, act, repeat.
- Fill in and submit a web form when no API exists, then screenshot the confirmation page.
<!-- END MANUAL -->

---
