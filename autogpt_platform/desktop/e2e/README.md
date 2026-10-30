# Installed-app tests

These tests take a finished installer, install it, and use the app the way a
person does: start it, create the first account, build and run an agent,
quit, start again, upgrade, uninstall. They are the proof that a build works
on a machine that never had the build tools. `build/smoke_test.py` covers
the runtime alone; nothing else covers the installer, the Electron shell or
an upgrade.

The app is started as its shortcut starts it, with one extra switch
(`--remote-debugging-port`) so that Playwright can attach to the app's own
window. No browser is downloaded.

| File | What it proves |
| --- | --- |
| `tests/01-first-run.spec.ts` | A silent install starts nothing. The first start reaches `ready`. The first account is the owner (admin in the session, on `/admin`, and to the backend), and a second sign-up is refused. An agent runs and reports over the websocket. Quitting leaves no process, changes nothing in the installed tree, and opens no socket the firewall would ask about. |
| `tests/02-relaunch.spec.ts` | The second start has the same address, account and data. Registration is still closed. An install that has an account but no admin gets its owner back on the next start. |
| `tests/03-upgrade.spec.ts` | A newer build installs over the old one (on Windows, while the app is running) and starts with the old data. |
| `tests/04-uninstall.spec.ts` | The program is removed and the data is kept. |

The files depend on each other and run in name order in one worker. Nothing
is retried. A failure ends the file it is in, not the run: a later file
fails at once with "01-first-run has to pass first" when what it needs was
never recorded, and runs when it was.

A quit is checked each time: no process left, the runtime stopped by itself
within the minute the shell allows before it kills it, and PostgreSQL removed
its `postmaster.pid`, which it does when it is shut down and not when it is
killed.

When a start fails, the report has the service logs and whatever the
launcher and the shell printed (`test-results/launcher/`). An app that exits
before its runtime starts fails the test within about half a minute instead
of at the timeout.

## Running

```
npm ci
npx playwright test
```

Set `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1` for `npm ci`. Node 20 or newer.

With no variables set, the tests use the app that is already installed in
its default location, skip the upgrade and do not uninstall it. The first
test needs a data directory with no database in it, so on a machine where
you use the app, point `AUTOGPT_E2E_DATA_DIR` at an empty directory. The app
is then also started with `--user-data-dir=<that directory>/electron-profile`,
so the tests' cookies and sign-outs stay out of the profile your own install
uses.

| Variable | Meaning |
| --- | --- |
| `AUTOGPT_E2E_INSTALLER` | The installer to test: a file, or a directory holding exactly one installer of the kind. When set, the app must not be installed yet, and the last test uninstalls it. |
| `AUTOGPT_E2E_UPGRADE_INSTALLER` | A build with a higher version, for the upgrade test. Skipped when unset. |
| `AUTOGPT_E2E_KIND` | `nsis`, `dmg`, `deb` or `appimage`. Default: `nsis` on Windows, `dmg` on macOS, `deb` on Linux. |
| `AUTOGPT_DESKTOP_VARIANT` | The slug the installers were built with, when they are a variant's (`../README.md`, "Variants"). The install location, the executable, the package and the data directory are then that variant's, from the same definition the build uses (`../src/identity.js`). Unset for the normal app. |
| `AUTOGPT_E2E_DATA_DIR` | Run the app on this data directory instead of its default one, with its own browser profile inside it. A variant uses the directory next to it, `<directory>-<slug>`, as it does for `AUTOGPT_DESKTOP_DATA_DIR`: the directory itself is the normal app's. |
| `AUTOGPT_E2E_INSTALL_DIR` | Where the app is installed, if not the default (Windows: the install directory; macOS: the `.app`; AppImage: the directory holding `AutoGPT.AppImage`, or `AutoGPT-<slug>.AppImage` for a variant). |
| `AUTOGPT_E2E_UNINSTALL` | `1` to let the last test uninstall an app these tests did not install. |
| `AUTOGPT_E2E_FIREWALL` | `1` to run the Windows firewall check. See below before setting it. |
| `AUTOGPT_E2E_FIRST_READY_SECONDS` | How long the first start may take. Default 600. |
| `AUTOGPT_E2E_READY_SECONDS` | How long a later start may take. Default 300. |

Per OS:

- **Windows.** `AutoGPT-Setup-*.exe /S` into `%LOCALAPPDATA%\Programs\autogpt`.
  The install takes minutes; how many is recorded (see below).
- **macOS.** The disk image is marked as downloaded, mounted, and the app is
  copied to `/Applications`. The test records what `spctl` says about it and
  then removes the quarantine mark, which is what "Open Anyway" does: the
  first-launch dialog cannot be answered without a person.
- **Linux, .deb.** `sudo apt-get install ./AutoGPT-*.deb`; needs
  passwordless `sudo`.
- **Linux, AppImage.** Copied to `~/Applications/AutoGPT.AppImage`.
  Installs `libfuse2` with `sudo` when it is missing.
- **Linux without a display** (CI): the app runs under
  `xvfb-run -a dbus-run-session --`, so `xvfb`, `xauth` and `dbus` must be
  installed. `xauth` is only recommended by `xvfb`, not required, and
  `xvfb-run` does not start without it.

## Durations

What a user waits for is timed on every run: the installer, the first start
to `ready`, each later start, the upgrade (the installer run over the old
version), and the first start after it. Each is a note on the test that
measured it (`seconds: install`, `seconds: first start, to ready`, ...), and
all of them are in `test-results/durations.md`, a table in the order they
happened:

```
| What | Seconds |
| --- | ---: |
| install | ... |
| first start, to ready | ... |
| restart, to ready | ... |
| upgrade (the installer, over the old version) | ... |
| first start after the upgrade, to ready | ... |
```

Timing an install means something only on a machine like a user's: a clean
one, with the virus scanner on.

On Windows the upgrade's installer is started from inside the install
directory, as an update starts it (from the running app, whose working
directory that is). Windows does not rename a directory a program is
standing in, and one rename is how the old version is moved away
(`resources/installer.nsh`): started from anywhere else, the upgrade timed
here would be one no user gets. The removal that is compiled into a version
runs when the next one replaces it, so the fast one is measured from the
second upgrade on.

## The firewall check

`AUTOGPT_E2E_FIREWALL=1` runs `lib/firewall.ps1`, which **changes the
machine**: it turns the firewall and its notifications on for every profile,
enlarges the Security log, and turns on auditing of every socket any program
opens. It needs an administrator. It is for CI machines that are thrown
away. On your own machine it would also show a firewall prompt for its
control program.

It then reads the audit log for the app's programs. A listener on anything
but loopback is a failure. Before trusting the log it runs a control: a
program that does listen on every interface must appear in it, or the test
fails as "instrumentation is not working".

## Known to fail today

- **macOS in CI.** The runner has 7 GB and the stack uses about 6. If the
  first start does not reach `ready` in time, the fix is in the app's memory
  use, not in this timeout.

- **"Running the app changed the install directory."** Expected until the
  frontend's image cache is kept out of the installed tree. The check is
  there to show when that is done, and to keep it done. It fails the last
  test of a file; the files after it still run.

## When upstream changes the frontend

The tests depend on these upstream names in the UI:

- the sign-up form: `Create your account`, the `Email` label, `#password`,
  `#confirmPassword`, the terms checkbox, `Sign up`;
- the login form: `Log in to your account to continue`, `Log in`;
- `/library` showing the agent's name;
- `/admin/marketplace`, an admin page: it must stay put for the owner, where
  anyone else is sent to `/`.

The proof that the first account is an admin also uses one admin-only
backend route, `GET /api/executions/admin/execution_analytics/config`, which
must answer 200. Any admin-only page and route would do if upstream renames
these two; both are in `tests/01-first-run.spec.ts`.

Everything else goes through the REST API and the websocket. The form
selectors are in `lib/session.ts`, the API calls in `lib/session.ts` and
`lib/agent.ts`.
