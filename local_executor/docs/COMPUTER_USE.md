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

Computer use is off by default. Install the package with the computer-use
extra, which brings the pinned cua-driver binary:

```bash
pipx install 'autogpt-local-executor[computer-use]'
```

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
`DISPLAY` (or `WAYLAND_DISPLAY`) and `DBUS_SESSION_BUS_ADDRESS` in its
environment, which desktop sessions usually import into the user manager.
`doctor` runs cua-driver's own checks for the display server and AT-SPI.

WSL2 can't drive Windows apps; run the shim on Windows itself.

## Clipboard

`--enable-clipboard` offers `clipboard_write` (text only) and
`clipboard_read`. What a read returns depends on a second setting:

- **Default:** only text the agent itself put on the clipboard in the last
  30 seconds. Anything you copied comes back as `CLIPBOARD_CONCEALED`.
- **With `--enable-clipboard-read-foreign`:** whatever is on the clipboard,
  except what a password manager marked private (many mark their copies,
  KeePassXC among them). Those come back as
  `CLIPBOARD_CONCEALED`. On Linux the shim checks with `xclip` (X11) or
  `wl-paste` (Wayland); without the tool it refuses rather than guess.

The agent can't put a file or image on the clipboard.

## What leaves this computer

Everything a tool returns goes to the platform and the model: accessibility
trees, which can include the text of windows the agent reads, and
screenshots. Each call, refused ones too, is recorded in the audit log,
without the text the agent typed or scripted (see
[AUDIT_LOG.md](AUDIT_LOG.md)). cua-driver's own telemetry is off for the
driver the shim starts.

## Troubleshooting

Start with `autogpt-shim doctor`. Then:

| Symptom | Cause |
|---|---|
| The platform doesn't offer computer use for this computer | The shim wasn't started with `--enable-computer-use`, cua-driver isn't installed, or it didn't start; the shim log says which (`Not advertising computer_use: ...`). On Linux, no display in the shim's environment. |
| The shim exits with code 78 at start | macOS permission missing, or no usable cua-driver. `doctor` says which. |
| `DEPENDENCY_MISSING` | No cua-driver, or one of another release. Reinstall with the computer-use extra. |
| Windows look empty to the agent (`degraded: true`) on Linux | AT-SPI unreachable: the session bus isn't in the shim's environment, or the app doesn't expose accessibility. |
| Results arrive without screenshots | The per-minute screenshot budget is spent; the result says so. |
| `FEATURE_NOT_SUPPORTED` | The agent asked for a tool this computer doesn't offer, or an argument it refuses, such as writing a screenshot to disk. |

## Protocol 2.0

Protocol 1.x had its own screen-coordinate operations (`SCREENSHOT_REQUEST`,
`INPUT_ACTION`, `WINDOW_LIST_REQUEST` and the rest) served by pyautogui,
mss, pyobjc and pywin32. Protocol 2.0 replaced them with cua-driver's tools;
a 2.0 shim and a 1.x platform refuse each other at the handshake (close code
4426), so update both.
