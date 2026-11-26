# ⚠️ EXPERIMENTAL — AutoGPT Local PC Executor

> **DANGER: experimental software that can read and write local files and,
> when explicitly enabled, execute unrestricted user-level shell commands on
> instruction from a cloud LLM.** Do not run on any machine you care about.
> Do not run as root. `--allowed-root` protects `FILE_*` operations only; it
> does not sandbox the shell.

> This package lives in the [AutoGPT repository](https://github.com/Significant-Gravitas/AutoGPT/tree/dev/local_executor),
> next to the platform it connects to. It was developed in
> `Significant-Gravitas/autogpt-local-executor` until then.

---

## What Is This?

A daemon you install on your local machine that connects it to the [AutoGPT hosted platform](https://platform.agpt.co) as an execution backend — instead of an E2B cloud sandbox.

Once connected, AutoGPT can:
- Read and write files on your filesystem (jailed to the folder selected for that chat)
- *(Optional, disabled by default)* Execute shell commands as your user
  (per-OS shell selection — bash/zsh/pwsh/cmd)
- *(Optional)* Operate apps on this computer through [cua-driver](https://github.com/trycua/cua): read windows, click, type, fill forms, in the background where it can
- Hardware access (serial, USB, GPIO) is a design, not a feature yet: see [HARDWARE.md](docs/HARDWARE.md).
- Local-model routing is disabled scaffolding; it is not a supported chat mode.

Cross-platform: macOS, Windows, Linux (Tier 1); WSL2 + Windows-on-ARM
(Tier 2). See [CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md) for the
support matrix.

## Platform Binding Layer

The platform-side code (WebSocket route, `LocalPCShim`, `ShimConnectionManager`, rollout flags) lives in the AutoGPT monorepo, where it merged with:
👉 https://github.com/Significant-Gravitas/AutoGPT/pull/13050

## Install (alpha)

Install the revision your deployment pins. The platform shows the command
on its Local PC setup card, in a new chat's folder dialog while no computer
is connected, and the revision it names is the one whose protocol that
deployment speaks:

```bash
# From the setup card, with its revision in place of <revision>:
pipx install "git+https://github.com/Significant-Gravitas/autogpt-local-executor.git@<revision>"
```

A shim of another revision may speak another protocol version than the
platform, and the two then refuse each other (close code 4426).

```bash
# One-time: sign in. With no --platform-url this is the hosted AutoGPT
# platform (see "Which platform" below).
autogpt-shim auth

# Keep the paired machine online. This opens an outbound control connection;
# the platform's new-chat flow lets you browse and select a folder per chat.
autogpt-shim start

# High-risk capabilities are explicit start-command opt-ins:
autogpt-shim start --enable-shell

# Compatibility only: connect directly to one pre-existing copilot session.
# In this legacy mode --allowed-root remains the one session's file jail.
autogpt-shim \
  --allowed-root ~/autogpt-workspace \
  start --session-id "<copilot-session-id>"

# --enable-recording is reserved for the design preview. The daemon currently
# refuses this flag with EX_CONFIG because capture + interpretation aren't yet
# safe to expose end to end.
```

Computer use needs a few more packages, added to the install you have
rather than by installing again: see
[COMPUTER_USE.md](docs/COMPUTER_USE.md#turning-it-on).

### For development

To run the package from your checkout of this repository, with your changes
in it, from the repository's root:

```bash
pipx install --editable ./local_executor
```

For work on the package itself, `AGENTS.md` beside this file sets up a
virtual environment with the test tools.

### Which platform

With nothing configured, the shim connects to the hosted AutoGPT platform:
its backend at `https://backend.agpt.co`, and its sign-in pages at
`https://platform.agpt.co/auth`. Those are the two URLs the platform's setup
card passes as `--platform-url` and `--platform-oauth-url`, so the card's
commands and the bare ones above use the same sign-in.

For a platform you host yourself, give its backend as `--platform-url` (or
`platform_url` in the config file) to every command. Its sign-in pages are
taken to be at `<platform-url>/auth`; where the frontend is on another host,
give their base as `--platform-oauth-url` too. The hosted platform's sign-in
pages are only ever the default for the hosted backend.

For autostart, run `autogpt-shim install`, enable the generated per-user service,
and omit `session_id` from the TOML configuration. The service maintains the
persistent paired-machine connection; selected roots are isolated per chat.
`autogpt-shim uninstall` stops that service, unregisters it and removes the
autostart entry; it exits 1, and prints the command to run, if it couldn't
stop the service (see [CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md#uninstalling)).
The service names a config file with `--config` when you passed one or a file
exists at the default location; otherwise the shim looks at the default
location itself. `install` refuses a `--config` it can't load, such as one that
names no file, since the service could never start with it; create the file
before you install. The service also gets the `--platform-url`,
`--platform-oauth-url`, `--allowed-root` and `--session-id` you give
`install`, or, for any you leave out, the `AUTOGPT_SHIM_` variable that sets
it in `install`'s environment: credentials are kept per deployment, so give
it the same platform URLs you gave `auth`. The service doesn't get any other
setting from that environment, so `install` refuses
`AUTOGPT_SHIM_PLATFORM_WS_URL` and `AUTOGPT_SHIM_OAUTH_CLIENT_ID` there (they
choose the deployment too; put them in the config file) and names any other
`AUTOGPT_SHIM_` setting the service won't see.

The platform never sends an arbitrary host path to the shim. Folder navigation
uses short-lived opaque references issued by the host, and selecting a folder
produces a machine-signed root grant that can restore that chat after a daemon
restart. Folder names and canonical paths are visible to the paired platform
during browsing; file names, file content, sizes, timestamps, and permissions
are not returned by this control API.

The daemon also fails startup if its tamper-evident audit writer cannot be
initialized; there is no implicit unaudited mode.

Then on the platform: ask an operator to flip the `local-pc-executor`
PostHog feature flag for your user. Once on, copilot turns route through
your shim instead of E2B. Audit log lives at the per-OS path documented
in [AUDIT_LOG.md](docs/AUDIT_LOG.md); review it with
`autogpt-shim audit tail` / `verify`.

### Several computers

One account can have a shim on each of several computers. The platform's
folder dialog lists each connected computer by its `display_name` and its
operating system. The display name is the computer's hostname unless you
set another, with `display_name = "Work laptop"` in the config file or
`AUTOGPT_SHIM_DISPLAY_NAME`; two computers with the same hostname look the
same in that list until one of them gets a name of its own.

Behind the name, the platform knows each computer by its `machine_id`: a
UUID the shim derives for itself from the operating system's own
identifier and a random value that the first `autogpt-shim start` makes
and keeps in `~/.autogpt-local-executor/machine-id-seed`. Neither of the
two is sent. The computer keeps its ID across restarts, upgrades and
reinstalls of the shim for as long as that file is kept. Delete it and the
platform sees a new computer, and each chat that was bound to the old one
asks for its computer and folder again. See
[CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md#machine-id).

### Recording design preview (disabled)

Recording protocol and macOS capture code exist for design/testing, but the
capture-to-interpretation path is incomplete. The daemon never advertises the
capability, and setting `enable_recording=true` or passing `--enable-recording`
fails startup clearly. The intended trust boundary below remains design
documentation, not a currently enabled product promise.

## Docs

| Doc | Description |
|-----|-------------|
| [VISION.md](docs/VISION.md) | Full dream + platform changes needed per capability |
| [PROTOCOL.md](docs/PROTOCOL.md) | WebSocket message protocol spec |
| [OAUTH_FLOW.md](docs/OAUTH_FLOW.md) | Auth design (uses AutoGPT's existing OAuth provider) |
| [SECURITY.md](docs/SECURITY.md) | Threat model and defense layers |
| [PLATFORM_HOOKS.md](docs/PLATFORM_HOOKS.md) | The platform-side design sketch, and what the platform built |
| [CROSS_PLATFORM.md](docs/CROSS_PLATFORM.md) | Per-OS support matrix and path-jail strategy |
| [AUDIT_LOG.md](docs/AUDIT_LOG.md) | Audit log format, HMAC tamper-evidence, CLI |
| [HARDWARE.md](docs/HARDWARE.md) | Hardware access wire spec (serial / USB / GPIO) |
| [COMPUTER_USE.md](docs/COMPUTER_USE.md) | Computer use: setup per OS, clipboard, privacy, troubleshooting |
| [CUA_DRIVER.md](docs/CUA_DRIVER.md) | How the shim runs cua-driver: tool catalog, policy, enforcement, residual risk |
| [MULTI_MACHINE.md](docs/MULTI_MACHINE.md) | Multi-machine orchestration spec (one session, N shims) |
| [PRIVACY_MODE.md](docs/PRIVACY_MODE.md) | Privacy mode design proposal — not a current guarantee that files or local-model content stay on the machine |
| [RELEASE_RUNBOOK.md](docs/RELEASE_RUNBOOK.md) | Operator runbook for PyPI / Homebrew / Scoop publish |
| [WORKFLOW_RECORDING.md](docs/WORKFLOW_RECORDING.md) | Record a workflow → agent generalizes it into a skill (design draft) |

## Contributing

Issues with `[local-executor]` prefix. PRs welcome — understand this interface will break repeatedly.
