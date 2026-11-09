# Computer use

With computer use, an AutoGPT agent operates apps on this computer: it
reads windows, clicks, types, fills forms, runs menus and drives Chromium
pages. The shim runs [cua-driver](https://github.com/trycua/cua) to do it and
relays a reviewed set of cua-driver's tools to the platform.
[CUA_DRIVER.md](CUA_DRIVER.md) explains the design and its limits;
[PROTOCOL.md](PROTOCOL.md#computer-use) has the wire messages.

The agent works mostly in the background. It reads a window's accessibility
tree and acts on its elements directly, so the window you are using keeps
focus, and it checks what changed. It falls back to clicking pixels, or to
bringing a window forward, only when an app gives it no other way. On Windows
and Linux, a cursor overlay in the session color shows where the agent is
working.

## Turning it on

Computer use is off by default. It needs the computer-use extra, which
brings the pinned cua-driver binary. A shim installed from the platform's
setup card or the README doesn't have it, and pipx leaves an installed
package as it is, so reinstall it with the extra from the same Git source:

```bash
pipx install --force "autogpt-local-executor[computer-use] @ git+https://github.com/Significant-Gravitas/autogpt-local-executor.git"
```

If your install command named a revision (`...autogpt-local-executor.git@<revision>`),
add the same `@<revision>` at the end of the URL here. The command works as
it is in Windows `cmd.exe` and PowerShell too: both take the double quotes.

Then start the shim with `--enable-computer-use` (or set
`enable_computer_use = true` in the config), and run `autogpt-shim doctor`.
The platform still has to grant the `computer_use` capability to each chat
session; the shim advertises it only once cua-driver has started.

| Setting | Default | Effect |
|---|---|---|
| `--enable-computer-use` | off | Offer computer use. |
| `--enable-clipboard` | off | Also offer `clipboard_read` and `clipboard_write`. |
| `--enable-clipboard-read-foreign` | off | Let `clipboard_read` return what you copied, not only what the agent wrote. See [Clipboard](#clipboard). |
| `max_screenshots_per_minute` | 10 | Screenshots sent to the platform per minute, across all sessions. |
| `cua_driver_path` | unset | A cua-driver to run instead of the bundled one. It must be the same minor release (0.38). |

## macOS

Computer use needs **Accessibility** (to act) and **Screen Recording** (to
see; without it a capture shows only the wallpaper and menu bar).

The shim runs its own cua-driver, so both permissions belong to the app that
runs the shim: the terminal you start it from, or, for the autostart service,
the Python that runs it. It doesn't use a CuaDriver.app you may have
installed for other tools (see [CUA_DRIVER.md](CUA_DRIVER.md#how-the-driver-runs)).

`autogpt-shim doctor` reports both permissions and asks macOS to show the
prompts for any that are missing. If computer use is on and a
permission is missing when the shim starts, it exits with code 78 and
launchd starts it again later; grant the permission in System Settings →
Privacy & Security and it connects on the next try.

## Windows

The shim must run in your desktop session (the installer's logon task does);
a service in session 0 has no desktop to drive. A shim that isn't elevated
can't drive the windows of apps running as Administrator. `doctor` says which
you have.

## Linux

Computer use needs a desktop session: an X11 display, or Wayland. On Wayland
cua-driver uses XWayland unless you set `CUA_DRIVER_RS_ENABLE_WAYLAND=1` in
the shim's environment for its native backend, whose support varies by
compositor (see cua-driver's Linux notes). The accessibility tree comes from
AT-SPI over the session bus: a shim started as a systemd user service needs
`DISPLAY` (XWayland's, on Wayland) and `DBUS_SESSION_BUS_ADDRESS` in its
environment, which desktop sessions usually import into the user manager.
`WAYLAND_DISPLAY` stands in for `DISPLAY` only with
`CUA_DRIVER_RS_ENABLE_WAYLAND=1`; without either, the shim doesn't offer
computer use. `doctor` runs cua-driver's own checks for the display server
and AT-SPI.

WSL2 can't drive Windows apps; run the shim on Windows itself.

## Clipboard

`--enable-clipboard` offers `clipboard_write` and `clipboard_read`. What a
read returns depends on a second setting:

- **Default:** only text the agent itself put on the clipboard in the last
  30 seconds, from any chat on this computer: the clipboard is the
  computer's, not one chat's. Anything else comes back as
  `CLIPBOARD_CONCEALED`. The shim compares the text, so a copy you make of
  the very same text reads back too.
- **With `--enable-clipboard-read-foreign`:** whatever is on the clipboard,
  except what a password manager marked private (many mark their copies,
  KeePassXC among them). Those come back as
  `CLIPBOARD_CONCEALED`. On Linux the shim checks with `xclip` (X11) or
  `wl-paste` (Wayland); without the tool it refuses rather than guess.

The agent can put a file or an image file on the clipboard only after you
approve that call (see [Files on the disk](#files-on-the-disk)).

## Files on the disk

Three of cua-driver's tool arguments reach this computer's disk themselves:
saving a screenshot to a file, putting a file or image file on the
clipboard, and handing files to a web page's upload field. Otto asks you
first, every time, on an approval card in the chat that shows the path; the
shim runs the call only once you approve it. Without an approval the shim
answers `APPROVAL_REQUIRED` and nothing happens.

A script (`run_script`) asks too, on a card that shows it: the
shim can't check the calls a script makes inside cua-driver, and one of
them can save a screenshot to a file.

So does a `launch_app` call that runs more than an app by its name: a
command line or a program's path, extra command-line arguments, links or
files to open, or an inspector port. cua-driver starts those out of sight,
with no window of their own, and opens a link in your own browser. Opening
an app by its name or id asks nothing.

In an Unsupervised chat, none of these ask: the platform approves the call
itself and tells the shim so, and the shim, which can't tell that from your
approval, runs it.

These approvals cover those arguments, not the apps. An app Otto drives
reads and writes files with your access, outside the shared folders too: a
terminal or a file manager, say. See
[CUA_DRIVER.md](CUA_DRIVER.md#residual-risk).

## What leaves this computer

Everything a tool returns goes to the platform and the model: accessibility
trees, which can include the text of windows the agent reads, and
screenshots. Each call the computer-use handler takes, refused ones too, is
recorded in the audit log, without the text the agent typed or scripted
(see [AUDIT_LOG.md](AUDIT_LOG.md)). Recording is best effort: a call goes
ahead if its record can't be written, and a call refused before it reaches
the handler (computer use off on this machine, or not granted to the
session) is answered without one. cua-driver's own telemetry is off for the
driver the shim starts.

## Troubleshooting

Start with `autogpt-shim doctor`. Then:

| Symptom | Cause |
|---|---|
| The platform doesn't offer computer use for this computer | The shim wasn't started with `--enable-computer-use`, cua-driver isn't installed, or it didn't start; the shim log says which (`Not advertising computer_use: ...`). On Linux, no display in the shim's environment. |
| The shim exits with code 78 at start | A macOS permission is missing, or there is no usable cua-driver: none installed, or on macOS one of another release. `doctor` says which. For a permission, launchd starts the shim again on its own. A missing or incompatible driver isn't retried: the LaunchAgent exits 0 instead, and the systemd unit doesn't restart on 78, so restart the service once the driver is fixed (`launchctl kickstart -k gui/$(id -u)/net.autogpt.shim`, or `systemctl --user restart autogpt-shim`). |
| `DEPENDENCY_MISSING` | No cua-driver, or one of another release. Reinstall with the computer-use extra. |
| Windows look empty to the agent (`degraded: true`) on Linux | AT-SPI unreachable: the session bus isn't in the shim's environment, or the app doesn't expose accessibility. |
| Results arrive without screenshots | The per-minute screenshot budget is spent; the result says so. |
| `FEATURE_NOT_SUPPORTED` | The agent asked for a tool this computer doesn't offer, or an argument it never accepts, such as `check_permissions` with `prompt: true`. |
| `APPROVAL_REQUIRED` | The call reaches the disk, or starts a program out of sight, and you haven't approved it. |
