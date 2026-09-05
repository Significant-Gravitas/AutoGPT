# Live desktop preview for Instantiate Code Sandbox

The existing block supports `enable_live_view=true` and returns `live_url`
alongside its normal outputs. The default remains disabled. Execute Code Step
continues using the returned `sandbox_id` and the same notebook interpreter.

## Build the template once

From `autogpt_platform/backend`, with `E2B_API_KEY` configured in the environment
or `.env`:

```sh
poetry run python scripts/e2b_desktop/build.py
```

This creates the `autogpt-code-desktop` template in that E2B account. Use
`--alias your-template-name` to choose another name. Template builds and running
sandboxes use your E2B account resources.

The build allocates 8 vCPUs and 8 GB RAM (8192 MB). Rebuild the template
after changing these settings and create a new sandbox to use the new allocation.
AutoGPT's application dependencies and services still need to be configured.

The template extends E2B's `code-interpreter-v1` with XFCE, a virtual display,
noVNC, Firefox, and an editor. The standard `desktop` template does not include
the interpreter required by these blocks. Custom templates must provide both.
The Desktop SDK starts the graphical session when the block creates a sandbox.

## Use the existing blocks

1. On Instantiate Code Sandbox, select credentials from the account containing
   the template, set `template_id=autogpt-code-desktop`, and enable live view.
2. Open its `live_url` to view and interact with the desktop. The link includes
   the stream password; anyone with it can control the desktop while it runs.
3. Pass `sandbox_id` to Execute Code Step for subsequent work. The ID is emitted
   after setup completes so dependent workflow steps cannot race setup.
4. Use the Bash language to launch visible applications, for example
   `DISPLAY=:0 mousepad /home/user/example.py >/tmp/editor.log 2>&1 &`.
   File writes and notebook execution do not appear as terminal typing.
5. For manual PR testing, have the agent start the app and its required services
   inside this sandbox and open the app in Firefox. Set `timeout=1800` on the
   final Execute Code Step for a 30-minute lifetime measured from that step's
   start, and leave `dispose_sandbox=false`. Each step's reconnect resets the
   lifetime, so set its timeout explicitly when you need more than E2B's default.
6. When finished testing, use Execute Code Step with `dispose_sandbox=true` to
   terminate it. The sandbox also expires at the configured timeout; the URL
   works only while it runs.

The block-created sandbox is separate from CoPilot's built-in session sandbox.
Experts must use Execute Code Step with this ID to work in the viewed sandbox.
This change does not add an embedded viewer or alter CoPilot's sandbox lifecycle.

Sources: [E2B Desktop](https://github.com/e2b-dev/desktop),
[code interpreter template](https://github.com/e2b-dev/code-interpreter/blob/main/template/template.py).
