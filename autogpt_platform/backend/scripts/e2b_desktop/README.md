# Live desktop preview for Instantiate Code Sandbox

The existing block supports `enable_live_view=true` and returns `live_url`
alongside its normal outputs. The default remains disabled. Execute Code Step
continues using the returned `sandbox_id` and the same notebook interpreter.

The backend must have `FRONTEND_BASE_URL` set to the public frontend URL so
preview links open through the authenticated frontend proxy.

## Build the template once

From `autogpt_platform/backend`, with `E2B_API_KEY` configured in the environment
or `.env`:

```sh
poetry run python scripts/e2b_desktop/build.py
```

This creates the `autogpt-code-desktop` template in that E2B account. Use
`--alias your-template-name` to choose another name. Template builds and running
sandboxes use your E2B account resources.

The build defaults to 8 vCPUs and 8 GB RAM (8192 MB). Set `--cpu-count` and
`--memory-mb` to change the allocation, then rebuild and create a new sandbox.
AutoGPT's application dependencies and services still need to be configured.

The template extends E2B's `code-interpreter-v1` with XFCE, a virtual display,
noVNC, Firefox, and an editor. The standard `desktop` template does not include
the interpreter required by these blocks. Custom templates must provide both.
The Desktop SDK starts the graphical session when the block creates a sandbox.
The SDK stays pinned to 2.3.0 because the bounded readiness adapter overrides its
retry behavior; revalidate that adapter before upgrading. Its dependencies require
Pillow >=11.1. noVNC and websockify are checked out at immutable commits.

## Agent responsibilities and interface

The workflow agent provisions the sandbox with Instantiate Code Sandbox, using
E2B credentials, the combined `template_id`, `enable_live_view=true`, a `timeout`,
and any `setup_commands` or `setup_code`. Successful setup returns `sandbox_id`
and the owner-authenticated `live_url`. The human owner opens `live_url`; the
agent and any collaborating experts pass `sandbox_id` to Execute Code Step with
`step_code` and `language` to work in that same environment. The workflow agent
extends the testing window with the step's `timeout` and disposes the sandbox
with `dispose_sandbox=true` when testing is finished.

## Use the existing blocks

1. On Instantiate Code Sandbox, select credentials from the account containing
   the template, set `template_id=autogpt-code-desktop`, and enable live view in
   advanced settings. Set `timeout` between 1 and 3600 seconds.
2. After setup succeeds, open `live_url` while signed in as the user who ran
   the block. The link stores an encrypted credential and checks ownership before
   redirecting; sharing run outputs does not grant desktop access. Links expire
   after 24 hours and require the sandbox to be running. The direct desktop URL
   after redirect includes the stream password; sharing that URL grants control.
3. Pass `sandbox_id` to Execute Code Step for subsequent work. Both the ID and preview link are emitted
   after setup completes so dependent workflow steps cannot race setup.
   Setup commands run on reconnects too; callers reusing them must make them idempotent.
4. Use the Bash language to launch visible applications, for example
   `DISPLAY=:0 mousepad /home/user/example.py >/tmp/editor.log 2>&1 &`.
   File writes and notebook execution do not appear as terminal typing.
5. For manual PR testing, have the agent start the app and its required services
   inside this sandbox and open the app in Firefox. Set `timeout=1800` on the
   final Execute Code Step to extend the remaining lifetime to at least 30 minutes
   when it connects, and leave `dispose_sandbox=false`. Reconnecting cannot shorten
   a longer remaining lifetime. Omitted timeouts use E2B's default extension.
6. When finished testing, use Execute Code Step with `dispose_sandbox=true` to
   terminate it. The sandbox also expires at the configured timeout; the URL
   works only while it runs.

The block-created sandbox is separate from CoPilot's built-in session sandbox.
Experts must use Execute Code Step with this ID to work in the viewed sandbox.
This change does not add an embedded viewer or alter CoPilot's sandbox lifecycle.

Sources: [E2B Desktop](https://github.com/e2b-dev/desktop),
[code interpreter template](https://github.com/e2b-dev/code-interpreter/blob/main/template/template.py).
