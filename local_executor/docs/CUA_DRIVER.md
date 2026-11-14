# Computer use through cua-driver

Computer use runs on [cua-driver](https://github.com/trycua/cua) (MIT), a
background GUI driver for macOS, Windows and Linux. The shim starts it, keeps
it to a reviewed set of tools, and relays those tools to the platform as they
are. The platform's agent then works the way cua-driver is built to be
driven: it picks a window, reads its accessibility tree, acts on elements by
token without taking focus, and checks the result, falling back to pixels or
foreground input only when it has to.

This replaced the shim's own backends (pyautogui, mss, pyobjc, pywin32),
which drove the whole screen by coordinates with foreground input and needed
a screenshot for every step.

[PROTOCOL.md](PROTOCOL.md#computer-use) has the wire messages and
[COMPUTER_USE.md](COMPUTER_USE.md) has setup and behavior per OS.

## How the driver runs

One `cua-driver mcp` process serves every session of a shim process. It
starts the first time it's needed, which is the first HELLO when computer
use is on, and stops with the shim. If it stops, the next call starts it
again, up to three starts in five minutes; after that, calls fail with the
driver's last error until the five minutes are up.

The shim speaks MCP to it over stdin and stdout
(`autogpt_local_executor/computer_use/mcp_client.py`): initialize, list
tools and call tools, one call at a time. cua-driver 0.38 runs one request
at a time and drops MCP's cancel notice, so a call can't be cancelled while
it runs: when one runs past its deadline, or its caller is cancelled, the
shim stops the driver to end it (that restart doesn't count toward the
three).

**Which binary.** `cua_driver_path` from the config, when it's set. If no
file is there, the shim doesn't fall back to another driver: with computer
use on, it doesn't start, as when no driver is installed. Without
`cua_driver_path`, the one in the `cua-driver` wheel the `computer-use` extra
installs (`cua-driver==0.38.0`), else `cua-driver` on PATH. It must be the
minor version this document reviews
(`REVIEWED_DRIVER_VERSION`); the shim refuses another, which could add tools
or arguments nobody has looked at.

**Which runtime.** The driver process the shim starts always owns the
runtime (`HELLO.computer_use_driver.runtime` is `direct`): it runs
`cua-driver mcp --direct` on every OS. Without `--direct`, cua-driver's
default on macOS relays to CuaDriver.app's service, and on Windows and Linux
it relays to a `cua-driver serve` daemon once the person has let the
driver's Computer History preview in (a preference file the driver reads at
start). The shim doesn't use that service:

- It is installed and updated by the person, not pinned with the shim, so it
  can be another release than the one this catalog was reviewed against,
  and the relay can't tell: the version it reports is the relay's own.
- It is shared with the person's other cua-driver clients, and runs with
  whatever permission mode and policy it was started with. The calls a
  script makes inside it wouldn't be bound by the shim's policy, and its
  telemetry follows its own setting.

So on macOS the Accessibility and Screen Recording grants belong to the app
that runs the shim (the terminal it was started from, or the Python that
runs the autostart service), as they did before cua-driver: the driver runs
as the shim's child and leaves prompting to its host. `autogpt-shim doctor`
requests them. The agent cursor overlay needs CuaDriver.app's AppKit host,
so it shows on Windows and Linux but not on macOS.

**Environment.** The driver gets the shim's environment (it and the apps it
launches need `DISPLAY`, the session bus and the rest), with these changes:

- The shim's own settings (`AUTOGPT_SHIM_*`, among them
  `AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE`) are dropped: nothing the driver starts
  needs them, and an app it launches would inherit them. That keeps them
  out of a terminal's `env`, not out of reach: any program running as the
  user can read the shim's own environment.
- Of cua's own settings (`CUA*`), only those cua's SDK hands a driver it
  embeds pass: `CUA_LOG`, `CUA_DRIVER_RS_HOME`, the accessibility and
  Wayland switches (`CUA_DRIVER_RS_ENABLE_WAYLAND`,
  `CUA_DRIVER_RS_DISABLE_A11Y_ADVERTISE`, `CUA_DRIVER_RS_A11Y_ADVERTISE_MODE`)
  and the pacing knobs (`CUA_DRIVER_WINDOW_CHANGE_TIMEOUT_MS`,
  `CUA_DRIVER_WINDOW_CHANGE_POLL_MS`, `CUA_DRIVER_KEY_GAP_MS`,
  `CUA_DRIVER_BATCH_PACING`), plus an administrator's managed policy
  (`CUA_DRIVER_MANAGED_POLICY_FILE`), which can only narrow what the driver
  allows. The rest are dropped: among them are switches that turn a browser
  sandbox or the key-focus guard off, an HTTP MCP port, the root of the
  driver's browser profiles, `CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS`, and
  capability manifests and session policies with their approvals (approving
  one is the launcher's decision, and this launcher doesn't make it).
- `CUA_DRIVER_PERMISSION_MODE=standard`, and `CUA_DRIVER_DISABLE_UNRESTRICTED=1`
  so nothing can switch it to `unrestricted`.
- `CUA_DRIVER_POLICY_FILE`: the shim's tool policy (below), in a private
  temporary directory that lives as long as the driver.
- `DO_NOT_TRACK=1`, `CUA_DRIVER_RS_TELEMETRY_ENABLED=false` and
  `CUA_DRIVER_RS_UPDATE_CHECK=false`. cua-driver sends content-free product
  telemetry by default; the shim's driver sends none and never checks for
  updates.

**Working folder.** On Linux the driver starts in an empty folder of its
own, a new one for each start, which nothing can be written to and which
the shim removes when it stops. The driver hands a `launch_app` `name`
that matches no installed app to `xdg-open` when a file or folder of that
name is in its working folder (below); there no name finds anything. The
apps the driver starts start there too, and a relative path in an approved
argument names nothing there, so give absolute paths. On macOS and Windows
the driver starts in the shim's own folder, from which it opens nothing by
name.

## What the platform gets

`COMPUTER_TOOLS_REQUEST` returns the driver's own instructions with the
shim's notes for this machine appended, and the offered tools with the
driver's descriptions, input schemas and annotations. `COMPUTER_TOOL_CALL`
relays one call and returns what the driver said, its refusals and failed
actions included, as `COMPUTER_TOOL_RESULT`. The platform should give the
model the instructions and let it call the tools by name; it should not
translate them into another action vocabulary.

## The catalog

`autogpt_local_executor/computer_use/catalog.py` lists what is offered,
reviewed against cua-driver 0.38.0:

| Group | Tools | Notes |
|---|---|---|
| Observe | `list_apps`, `list_windows`, `get_window_state`, `get_desktop_state`, `zoom`, `verify_state`, `get_screen_size`, `get_cursor_position`, `check_permissions`, `get_guide` | Read-only. `check_permissions` only reports. |
| Act | `launch_app`, `click`, `double_click`, `right_click`, `type_text`, `press_key`, `hotkey`, `scroll`, `drag`, `set_value`, `invoke_menu`, `set_window_frame`, `bring_to_front`, `move_cursor`, `mouse_button_down`, `mouse_button_up`, `mouse_drag`, `run_steps`, `run_script` | `run_steps` batches input; `run_script` runs sandboxed JavaScript whose only API is the driver's, and every script needs approval (below). On macOS a `click` that saves a screenshot (`debug_image_out`) needs approval too. |
| Browser | `get_browser_state`, `browser_prepare`, `browser_navigate`, `browser_click`, `browser_type`, `browser_select`, `browser_pointer`, `browser_dialog`, `browser_set_input_files` | Typed control of Chromium pages. In `standard` mode, without a grant, which the shim never gives, the driver reaches only a browser it launched with its own profile, never the person's signed-in one. `browser_set_input_files` hands files from the disk to a page, so every call needs approval (below). |
| Clipboard | `clipboard_read`, `clipboard_write` | Only with `--enable-clipboard`; see [Clipboard](#clipboard). |

Not offered:

| Tool | Why |
|---|---|
| `get_config`, `set_config`, `check_for_update`, `install_extension`, `install_ffmpeg` | The driver's configuration and installation belong to the person at the machine. |
| `start_session`, `end_session`, `get_session`, `get_session_state`, `list_sessions`, `escalate_session` | The shim owns sessions: it sets one label per shim session. |
| `set_agent_cursor_enabled`, `set_agent_cursor_motion`, `set_agent_cursor_theme`, `get_agent_cursor_state` | The cursor overlay shows the person where the agent is working; the agent may not hide or restyle it. |
| `start_recording`, `stop_recording`, `get_recording_state`, `replay_trajectory` | Write and read trajectories on disk. |
| `kill_app` | Ends processes. |
| `browser_download` | Writes downloads to the disk, and needs cua-driver's own host approval, which the shim doesn't relay yet. |
| `page` | Runs arbitrary JavaScript in pages (cua-driver allows it only in `unrestricted` mode anyway). |
| `parse_visual_regions` | Needs the `cua-perception` extension, which is AGPL. Never install it with the shim. |
| `parallel_mouse_drag`, `get_accessibility_tree`, `health_report` | Not needed: `drag` and `get_window_state` cover them, and `autogpt-shim doctor` reports health. |
| `debug_window_info` (Windows only) | A diagnostic for cua-driver's developers: `list_windows` and `get_window_state` cover what the agent needs. |

## Enforcement

Two layers, so neither is the only one.

**The driver.** The policy file allows exactly the offered tools
(`allow.tools`). cua-driver then lists only those, and refuses any other
call, including a call that `run_steps` or `run_script` makes inside the
driver, where the shim can't look. Those are narrower still: a `run_steps`
step can only be one of ten input tools, and a script can only reach
sixteen tools; all of them are in the catalog, which
`tests/test_cua_driver_contract.py` checks against the real driver.

**The shim** (`handlers.ComputerUseHandler`):

- Only an offered tool is relayed (`FEATURE_NOT_SUPPORTED` otherwise).
- Arguments that reach the disk need approval, call by call:
  `screenshot_out_file` on `get_window_state` and `get_desktop_state`
  (writes the screenshot to a file), `debug_image_out` on macOS `click`
  (writes a screenshot to a file before it clicks), `file_path` and
  `image_path` on `clipboard_write` (puts a file from anywhere on disk on
  the clipboard),
  `files` on `browser_set_input_files` (hands files to a page), and
  `script` on `run_script`, so every script: its calls run inside the
  driver, where the shim can't check them, and one can save a screenshot to
  a file or start any program. So do `launch_app`'s `launch_path`, `path`,
  `additional_arguments`, `urls` and `webkit_inspector_port`, which start a
  program out of sight (see Residual risk); `name`, `bundle_id` and `aumid`
  must name an app. The tool list names them in each tool's `approval_arguments`. The platform
  asks the person on an approval card that shows the path, and only then
  sends the call with the argument in `approved_arguments`; a chat that runs
  Unsupervised sends it without asking, and so does a standing rule the
  person set on an earlier card ("from now on"), for later calls of that
  kind whatever path they name. The shim, which can't tell these apart,
  trusts the field. Without it the shim answers
  `APPROVAL_REQUIRED` and nothing reaches the driver. A
  `run_steps` step may not use one at all, whatever tool it names (the
  driver runs a `press` with a target, and a `triple_click`, as a click),
  and neither may the `observe` that ends a batch, which the driver runs as
  a `get_window_state` call. Nor may a step be a `launch_app`, whose name
  the shim checks against the installed apps only when it's called on its
  own; cua-driver 0.38 batches no launch anyway. A batch the shim can't
  look inside is refused: `steps` that isn't a list, or a step that isn't
  an object.
- `prompt: true` on `check_permissions` is refused: permission prompts are
  raised by the person, with `autogpt-shim doctor`.
- `session` is set to the shim session's label on every call that takes it,
  overriding the caller's.
- The clipboard rules, the screenshot budget and the frame limit below.
- One call at a time per machine, and one deadline per call (120 s unless
  the platform asks for 1-600 s), counted from when the shim receives it, so
  the wait for its turn counts, and so does the wait for the driver to start.
  A call still waiting at its deadline never reaches the driver (`started:
  false`); a start it waited for goes on for the calls after it, since
  stopping one part way would count toward the three restarts. Past the deadline of a running
  call, the shim stops the driver (`started: true`), and the next call
  starts a new one; its sessions, element tokens and launched browsers'
  pages start over too.
- An audit record, best effort, for every call the computer-use handler
  takes, refused ones too: a call goes ahead if its record can't be
  written, and a call refused before the handler (computer use off on this
  machine, or not granted to the session) gets no record. An approved
  call's record names the approved arguments and keeps their paths whole.

**Why a policy and not a capability manifest.** cua-driver's capability
manifest is its strongest control, but it makes every window, app, display
and file path a declared resource: with a manifest loaded, observing or
acting on anything undeclared is refused, in `standard` mode too. That suits
a task whose apps are known in advance. Computer use here is open-ended, so
the manifest would have to declare everything, or the shim would need a
per-app consent step it doesn't have. The policy limits which tools exist
without limiting where they act. A per-app grant on top, written into a
manifest, is the natural next step.

## Residual risk

Computer use acts with the person's own rights through their GUI. An agent
that can type can open a terminal. The shim doesn't claim otherwise: the
capability is off until the person turns it on, the platform must grant it
to each session, the calls it takes are audited (best effort, see Audit),
and the agent cursor shows where the agent works. The approval arguments keep disk access out of side channels the
person can't see: each one waits for the person to approve it, unless the
chat runs Unsupervised or a standing rule the person set on an earlier card
("from now on") approves later calls of that kind, whatever their
arguments (see Enforcement). They are not a sandbox around the GUI.

Known gaps:

- A `run_script` script can pass `screenshot_out_file` to
  `get_window_state` inside the driver, writing a PNG where the script says
  (on macOS `click`'s `debug_image_out` too), and can call `launch_app`,
  which starts any program, unchecked.
  cua-driver 0.38's policy can allow a tool but can't forbid one of its
  arguments, and the shim can't see into scripts, so every script needs
  approval instead: a card that shows it, unless the chat runs Unsupervised
  or a standing rule set on an earlier `run_script` card approves later
  scripts, whatever they hold, unasked. A policy rule that forbids an
  argument belongs upstream; with one that kept scripts from those
  arguments and from `launch_app`, scripts could run unasked safely.
- `launch_app` can start any program with any arguments: `launch_path` (on
  Linux the `Exec=` line of a `.desktop` file, run split at spaces), a
  Windows `path`, and `additional_arguments`, where a Chromium flag such as
  `--renderer-cmd-prefix` runs a command. `urls` open in whatever handles
  them, a web page in the person's own signed-in browser. The driver starts
  all of these in the background, with no window of their own, so unlike a
  terminal the agent types into, nobody sees them: each is an approval
  argument. On Linux the driver also runs `name` as a command line before
  it looks for an app of that name, so the shim refuses a `name` whose first
  word is a program, a program's own name included (`reboot` would run it;
  an app goes by the name `list_apps` shows, such as `Firefox`). A `name`
  no installed app matches it then hands to `xdg-open` if a file or folder
  of that name is in its working folder (in the person's home, where it ran
  under systemd, `.bash_history` would open in an editor, for a screenshot
  to read, past the path jail and the approval `urls` needs). So the driver
  runs in an empty folder of its own (above), and where it acts on
  the `name` (no `urls`, no `launch_path`) the shim lets it through only
  when it is an installed app's name or desktop-file id, as `list_apps`
  lists them, ignoring ASCII case as the driver does. The driver's looser
  matches, part of a name or a launcher's program, don't count: another
  app installed meanwhile can leave them matching nothing. On Windows
  the driver hands a `name` that matches no installed app, and a
  `bundle_id` that isn't an AUMID (`Package!App`), to ShellExecuteEx, which
  runs a program of that name from PATH (`logoff` would sign the person
  out), so the shim refuses such a `bundle_id`, and lets a `name` through
  only when an installed app, as `list_apps` lists it, would match it the
  way the driver matches: the exact name, ignoring case and a trailing
  `.exe`, or one that starts with it. Either way it lists the installed
  apps for each launch it checks, just before the launch, so an app removed
  since an earlier listing doesn't let its name through. On every OS it refuses a `name`,
  `bundle_id` or `aumid` with a path, a URL scheme or a leading `-`. A plain launch, an app by its name or
  id, asks nothing, and a launched app can do whatever it does.
- `set_value` writes a field's value directly, without visible keystrokes.
  `type_text` does the same in the background through the accessibility
  API; the shim audits both.

## Clipboard

Off unless `--enable-clipboard`. Then `clipboard_write` takes exactly one
of `text`, `file_path` or `image_path`, the last two only once the call is
approved: by the person on its card, by a standing rule they set on an
earlier one, or in an Unsupervised chat (see Enforcement). `clipboard_read`
returns:

- by default, only text this shim wrote in the last 30 seconds, for any of
  its sessions: the clipboard is the computer's, so one record serves every
  chat on the machine. The shim keeps the hash of its last text write and
  compares it with what the driver reads, so the same text copied by the
  person matches too. Anything else is `CLIPBOARD_CONCEALED`
  (`writeback_only`, or `writeback_overwritten` if something replaced the
  shim's text, a file or image the shim put there included).
- with `--enable-clipboard-read-foreign`, whatever is there, except content a
  password manager marked private: macOS `org.nspasteboard.ConcealedType`,
  `TransientType` and `AutoGeneratedType`; Windows `CF_PRIVATE` formats and
  the `ExcludeClipboardContentFromMonitorProcessing` and
  `Clipboard Viewer Ignore` formats; Linux `x-kde-passwordManagerHint`
  (KeePassXC sets it too). cua-driver reads the clipboard without looking for
  these, so the shim checks the OS before the read
  (`computer_use/concealed.py`: AppKit, the Win32 API through ctypes, or
  `xclip`/`wl-paste`), checks the types the driver returned with the text,
  and checks the OS again after; a change counter that moved in between
  (macOS, Windows) refuses the read as `changed`. Linux keeps no such
  counter, and the driver reads the types and then the text, so a copy
  between the two would pass with another copy's types: there the shim
  reads through the driver a second time, and anything but the same text
  with no marker is refused as `changed`. A clipboard whose types can't be
  read counts as marked.

The text of a refused read never leaves the shim.

## Screenshots

Images come back inline, as the driver encoded them. At most
`max_screenshots_per_minute` (default 10) are sent a minute, across every
session on the machine; past that, results arrive without their images and
with a note saying so. The accessibility tree is the primary observation in
cua-driver's loop, so this budget rarely binds. A result that would exceed
the WebSocket frame limit loses its images first, then its structured
content, with a note either way.

## Audit

Every `COMPUTER_TOOL_CALL` the computer-use handler takes writes a
`COMPUTER_TOOL_CALL` record, best effort: a failed write doesn't stop the
call, and a call refused before the handler (computer use off, or not
granted to the session) has none. The record holds the tool,
the arguments with typed, set and scripted text (`text`, `value`, `script`,
`prompt_text`, a single character pressed with `press_key`, and the character
of a `hotkey` chord that types one: one that only holds Shift, or on macOS
also Option, or on Windows Ctrl with Alt, which Windows reads as AltGr. Only
names the driver holds as modifiers count, matched lowercased but not
trimmed as it matches them, so `[" shift ", "a"]` and `["capslock", "a"]`
hold none and type their character; a chord written as one string,
`press_key` key `shift+a` or `hotkey` keys `shift+a`, which the driver
splits at `+`, is redacted the same way)
replaced by its length and other long strings cut short, the
outcome, whether the driver reported an error, how many images were sent or
withheld, and the duration. A call cancelled by revocation is recorded as
`CANCELLED`: the shim stopped the driver to end it, but an action it
already made stays made. See [AUDIT_LOG.md](AUDIT_LOG.md).

## Permissions and preflight

With computer use on, the daemon checks before it connects that a usable
cua-driver is installed. If not, the failure is permanent: the shim exits 78,
which the systemd unit doesn't restart, and under the LaunchAgent it exits 0
so launchd doesn't restart it either. Once the driver is installed, restart
the service (`launchctl kickstart -k gui/$(id -u)/net.autogpt.shim`, or
`systemctl --user restart autogpt-shim`). The look for the driver gets 5
seconds (a chat's session, 2): a folder on PATH, or one Python imports from,
on a mount that isn't answering holds it, and the shim wouldn't connect at
all, file and shell work included. Past that the session connects without
the rest of this check, and the look goes on for HELLO, which offers
computer use only once the look finds the driver, and on macOS only once a
session's check of the grants below has passed (the next chat's, say, or the
next start's). On macOS it starts the driver,
where one of another release fails the same way, and asks
`check_permissions` for Accessibility and Screen Recording without
prompting; if either is missing it exits 78 and launchd starts it again
later, as before. Each chat's session checks again when it starts,
through the same driver. The platform gives a new chat's session a few
seconds to connect, so the check, its run as well as its start, ends
within two of them: if another chat's call holds the driver that long, or
the driver takes the check and doesn't answer, the new session goes ahead
unchecked, without waiting for a driver that overran the check to be
stopped. The grants are the app's, checked when the shim started, and a
busy driver is a working one.
Its HELLO waits as briefly for the screen size, the call's run as well as
its start, and leaves it out instead, without waiting for a driver that
overran the call to be stopped (the stop keeps the driver's turn, so no
call starts another driver until the old one is gone),
and as briefly for a driver that is starting again (after another chat's
call overran, say), and then lists the tools the driver offered when it
last ran, which its next start offers again.
`autogpt-shim doctor` runs cua-driver's own
`doctor`, starts the driver as the daemon does, reads the screen size, and on
macOS reports both grants and, when one is missing, asks macOS to show its
prompt for the shim's app (the driver leaves prompting to its host).

## Updating the driver

1. Bump the pin in `pyproject.toml` and `REVIEWED_DRIVER_VERSION`.
2. Diff the new release's `tools/list` against the old: new tools stay off
   until reviewed; new arguments on offered tools need the same review as
   the approval arguments and refusals above, file paths especially, on
   every OS: macOS's `click.debug_image_out` was missed in a review of the
   Linux list. The contract tests fail on any tool the catalog doesn't
   name, offered or in `NOT_OFFERED_TOOLS`, and, on the OS they run on, on
   an offered tool's argument named for a path (a `path`, `file`, `dir` or
   `out` word) that isn't an approval argument.
3. Check the new release's script and batch allowlists, its policy format,
   and its permission modes. If its policy can forbid an argument, forbid
   `screenshot_out_file` and `debug_image_out` inside scripts, and, once
   scripts can't reach `launch_app` either, drop `run_script` from
   `APPROVAL_ARGUMENTS`.
4. Run `tests/test_cua_driver_contract.py` on macOS, Windows and Linux; CI
   does.
5. Update this document's tables.
