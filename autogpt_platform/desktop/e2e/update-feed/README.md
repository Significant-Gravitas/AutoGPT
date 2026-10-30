# Real GitHub feed A → B proof

This is a separate, opt-in test. Ordinary installed-app tests still disable
updates and use their existing installer-over-install upgrade test. Run this
only after A and B have been published by the reviewed release workflow on
`ntindle/autogpt`. It never pushes, publishes, changes a feed, or supplies a
GitHub token to the installed app.

Use a fresh dedicated `variant/<slug>` with no installation, data, or pending
update cache on the test machine. The normal app requires a disposable machine
and `AUTOGPT_E2E_FEED_DISPOSABLE_NORMAL=1`. The test installs A into the app's
default location, creates a test owner and agent, and leaves the installation
and data available for inspection. Teardown stops remaining test processes;
that final teardown is not evidence of a graceful B shutdown.

The test invokes the packaged app's actual **Check for updates** and **Restart
to update** menu handlers through Playwright's Electron main-process API. It
answers only the exact idle-update confirmation. It does not replace the
updater, intercept network traffic, or install B itself. It checks the actual
download against an independently supplied SHA-256, then requires automatic
restart with the same public address, saved owner credentials, workspace file,
agent and completed execution, and verifies every recorded A process exited.
The owner password is kept in memory and omitted from the proof attachment.

Set all of these explicitly (example names; use real released versions, full
source SHAs and independently verified artifact hashes):

```text
AUTOGPT_E2E_REAL_FEED=1
AUTOGPT_DESKTOP_UPDATES=on
AUTOGPT_DESKTOP_VARIANT=feed-proof
AUTOGPT_E2E_KIND=nsis
AUTOGPT_E2E_INSTALLER=/path/to/AutoGPT-feed-proof-Setup-1.0.0-x64.exe
AUTOGPT_E2E_FEED_FROM_VERSION=1.0.0
AUTOGPT_E2E_FEED_TO_VERSION=1.0.1
AUTOGPT_E2E_FEED_FROM_SOURCE=<40 lowercase hex characters>
AUTOGPT_E2E_FEED_TO_SOURCE=<40 lowercase hex characters>
AUTOGPT_E2E_FEED_FROM_SHA256=<64 lowercase hex characters>
AUTOGPT_E2E_FEED_TO_SHA256=<64 lowercase hex characters>
AUTOGPT_E2E_FEED_TO_ARTIFACT=AutoGPT-feed-proof-Setup-1.0.1-x64.exe
```

From `desktop/e2e`, with its normal dependencies installed:

```sh
npx tsc -p update-feed/tsconfig.json
npx playwright test --config update-feed.unit.config.ts
npx playwright test --config update-feed.config.ts
```

On headless Linux run the final command under `xvfb-run -a dbus-run-session --`.
Use `appimage` for Linux self-update; `.deb` is notify-only. On macOS use `dmg`
for the A installer, but name and hash B's **arm64.zip** update payload. Both
Mac versions must carry the same Developer ID and pass notarization; ad-hoc
Mac builds deliberately cannot pass an automatic-update proof. Windows tests
need the interactive installer session used by `quitAndInstall(false, true)`.

Do not set runtime, data or install-directory overrides. Automatic OS relaunch
does not promise to retain debugging switches or ad-hoc environment overrides.
B is verified through its real HTTP API after its own automatic restart; this
test does not silently relaunch B to make the test pass. Version evidence is
the installed Windows product version, macOS bundle version, or the exact
replaced AppImage hash. A's version is read from both Electron and its browser.

The prerequisite check reads the real public GitHub release list, exact tags,
release notes, asset digests and B's `SHA256SUMS`. Both source pins must occur in
their release notes. Those notes identify the workflow's selected product
source; they are not independent proof that binary contents match source.
Review the release run and verify its artifact attestations before supplying
the pins. GitHub's automatic source archive currently follows the **ops tag**,
not the product SHA in the notes.

A dedicated variant proves its generic GitHub release-asset feed and isolates
the experiment from normal installs. It does **not** prove the normal app's
GitHub latest-provider path. A normal-app proof requires B to be the repository's
public latest release, which offers B to every eligible normal installation.
Drafts and `X.Y.Z-rc.N` versions are deliberately unsuitable. Run each platform
separately, keep the A/B proof attachment and logs, and do not infer Mac signing
or Windows installer behavior from a passing AppImage proof.
