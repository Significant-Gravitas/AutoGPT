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

Computer use is off by default. It needs the packages of the computer-use
extra: the pinned cua-driver binary, and on macOS the three pyobjc packages
the permission checks use. A shim installed from the platform's setup card
doesn't have them. Add them to the install you have. On Linux and Windows:

```bash
pipx inject autogpt-local-executor "cua-driver==0.38.0"
```

On macOS:

```bash
pipx inject autogpt-local-executor "cua-driver==0.38.0" "pyobjc-framework-ApplicationServices>=10.0" "pyobjc-framework-Cocoa>=10.0" "pyobjc-framework-Quartz>=10.0"
```

`autogpt-shim doctor` and the shim's own errors print the one for the
computer they run on. `pipx inject` installs the packages into the
environment pipx keeps for the shim, and records them as added to it. pip
leaves an installed package as it is, whatever it was installed from, so
the shim stays at the revision your deployment pins. What pipx recorded
the shim was installed from stays too: a later `pipx reinstall` installs
the shim from that record and adds the same packages back.

Two ways that look simpler change what is installed, now or later:

- Adding the extra by name (`pipx runpip autogpt-local-executor install
  "autogpt-local-executor[computer-use]"`) leaves the shim's files alone,
  but pipx then takes `autogpt-local-executor[computer-use]` for what the
  shim was installed from. Its next `pipx reinstall` or `pipx upgrade`
  looks for the shim by that name on the package index, not at the
  revision you installed.
- Reinstalling from the repository with no revision takes the default
  branch's head, which may speak another protocol version than your
  deployment. (`pipx install` without `--force` does nothing for a package
  pipx already has.)

In an install pipx doesn't manage, install the same packages with the pip
of the Python the shim runs under (`python -m pip install` and the quoted
names above). The commands work as they are in Windows `cmd.exe` and
PowerShell too: both take the double quotes.

Then turn computer use on in the config, `enable_computer_use = true` (or
`AUTOGPT_SHIM_ENABLE_COMPUTER_USE=true` in the environment), and run
`autogpt-shim doctor`. Doctor reads only those: `--enable-computer-use` is a
flag of `start`, so with the flag alone doctor skips the driver, display
and permission checks (it reports computer use "not requested in config")
and can pass. The platform still has to grant the `computer_use` capability
to each chat session; the shim advertises it only once cua-driver has
started.

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
launchd starts it again, every ten seconds or so, each time checking again;
grant the permission in System Settings → Privacy & Security and it
connects on the next try.

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
AT-SPI over the session bus, which a systemd user service already has.
`WAYLAND_DISPLAY` stands in for `DISPLAY` only with
`CUA_DRIVER_RS_ENABLE_WAYLAND=1`; without either, the shim doesn't offer
computer use. `doctor` runs cua-driver's own checks for the display server
and AT-SPI.

The systemd user service starts at login, before the desktop has exported
its display to the user manager, and a service keeps the environment it
started with. So when the service starts without `DISPLAY` or
`WAYLAND_DISPLAY`, the shim reads them from the manager
(`systemctl --user show-environment`, at most every 10 seconds) when it
decides whether to offer computer use and before each call, together with
`XAUTHORITY` and the other variables cua-driver reads (`XDG_SESSION_TYPE`,
`XDG_CURRENT_DESKTOP`, `XDG_SESSION_DESKTOP`, `DESKTOP_SESSION`, `SWAYSOCK`,
`HYPRLAND_INSTANCE_SIGNATURE`). It follows them when a new login brings a
new display (on GNOME, a new `XAUTHORITY` too) and restarts cua-driver with
them, and lets them go when the session ends. One read runs at a time, in a
thread of its own. While one stalls (`systemctl` looked up on a stalled
`PATH` mount, say), the shim waits for it 3 seconds at most when it decides
whether to offer computer use, and again when it starts cua-driver, then
goes on with the display as it is; a call waits for it until its own
deadline (`COMMAND_TIMEOUT` with `started: false`). The session's values
stand in for any the service started with, such as an `XAUTHORITY` an earlier session
left, which come back once it ends. A session without an `XAUTHORITY` takes
the service's away while it lasts, so Xlib reads `~/.Xauthority`, as it does
for the session's own programs. GNOME and KDE export them, as
do the Xsession and xinit scripts of most distributions. A session that
doesn't, such as some standalone window managers, needs
`dbus-update-activation-environment --systemd DISPLAY XAUTHORITY` (or
`WAYLAND_DISPLAY`) in its startup. HELLO is the only place the platform
learns that computer use is on offer, so a connection whose HELLO came
before the display (the service's, at login) closes once the display is up
and nothing is under way on it (on the machine's connection, nothing in
any chat either), and the reconnect's HELLO offers computer use. A shim
started by hand, or a service that started with a display, keeps its own.
The service has
`AUTOGPT_SHIM_SUPERVISOR=systemd` in its unit; an install from before
2026-11-09 picks it up with `autogpt-shim install`.

Apps the agent opens start in your home folder, as they do from your
desktop's launcher, whether the shim runs as the service or you started it
from a terminal in another folder: a terminal the agent opens is in your
home folder, and a save dialog starts there. See
[CUA_DRIVER.md](CUA_DRIVER.md#how-the-driver-runs).

The shim doesn't offer computer use under WSL2 at all, not even for Linux
apps WSLg shows; to drive your apps, run the shim on Windows itself.

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

The agent can put a file or an image file on the clipboard only once that
call is approved: by you on its card, by a standing rule you set on one, or
by the platform in an Unsupervised chat (see
[Files on the disk](#files-on-the-disk)).

## Files on the disk

Three of cua-driver's tool arguments reach this computer's disk themselves:
saving a screenshot to a file, putting a file or image file on the
clipboard, and handing files to a web page's upload field. Otto asks you
first, on an approval card in the chat that shows the path; the shim runs
the call only once it's approved. Without an approval the shim answers
`APPROVAL_REQUIRED` and nothing happens.

A script (`run_script`) asks too, on a card that shows it: the
shim can't check the calls a script makes inside cua-driver, and one of
them can save a screenshot to a file or start a program.

So does a `launch_app` call that runs more than an app by its name: a
command line or a program's path, extra command-line arguments, links or
files to open, or an inspector port. cua-driver starts those out of sight,
with no window of their own, and opens a link in your own browser. Opening
an app by its name or id asks nothing.

Two things approve these calls without a card. In an Unsupervised chat none
of them ask. And an approval card can also set a standing rule ("from now
on", in the scope the card names): later calls of that kind then go ahead
without a card, whatever path, file or script they name, not just the one
the card showed. Either way the platform approves the call and tells the
shim so, and the shim, which can't tell that from your approval of that one
call, runs it.

The browser tools open web addresses only: pages at `http://` and
`https://` addresses, and `about:blank`. Otto can't point cua-driver's
browser at a file on this computer (`file:///...`), at a page's source, or
at one of the browser's own pages: the shim refuses such an address before
anything happens, and records the refusal. To open a file or another kind
of link, Otto uses `launch_app` with links to open, which asks you first,
as above.

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
| The platform doesn't offer computer use for this computer | The shim wasn't started with `--enable-computer-use`, cua-driver isn't installed, or it didn't start; the shim log says which (`Not advertising computer_use: ...`). On Linux, no display in the shim's environment. Or looking for the driver stalled (`Looking for cua-driver took longer than ...`): a folder on PATH, or one Python imports from, may be on a mount that isn't answering. |
| The shim exits with code 78 at start | A macOS permission is missing, or there is no usable cua-driver: none installed, or on macOS one of another release. `doctor` says which. For a permission, launchd starts the shim again on its own. A missing or incompatible driver isn't retried: the LaunchAgent exits 0 instead, and the systemd unit doesn't restart on 78, so restart the service once the driver is fixed (`launchctl kickstart -k gui/$(id -u)/co.agpt.shim`, or `systemctl --user restart autogpt-shim`). |
| `DEPENDENCY_MISSING` | No cua-driver, or one of another release. Add computer use's packages to the install ("Turning it on" above). With `cua_driver_path` set, the shim runs the driver that setting names and no other: the packages don't change which one that is, so correct the setting, or take it out to use the packages' driver. |
| Windows look empty to the agent (`degraded: true`) on Linux | AT-SPI unreachable: the session bus isn't in the shim's environment, or the app doesn't expose accessibility. |
| Results arrive without screenshots | The per-minute screenshot budget is spent; the result says so. |
| `FEATURE_NOT_SUPPORTED` | The agent asked for a tool this computer doesn't offer, or an argument it never accepts, such as `check_permissions` with `prompt: true`. |
| `APPROVAL_REQUIRED` | The call reaches the disk, or starts a program out of sight, and you haven't approved it. |
