# Experimental native builds

Run from `autogpt_platform/desktop` in a disposable build checkout. The build
writes the frontend's `node_modules`, generated API clients and `.next`, its
private cache, and the output runtime. Never point the output or cache at an
installed application, another checkout, or a user profile.

Use Python 3.13, uv, Node 24, and the pnpm version in
`../frontend/package.json`. On Windows, compile Valkey with
`valkey-windows.sh` under MSYS2 before assembling the runtime. Its version and
source SHA-256 are in `artifacts.py`; the experimental CI workflow shows the
invocation. Do not use the Redis stand-in for distributable builds.

```sh
python -B build/build_runtime.py
npm ci
npm run dist -- --publish never
```

The dependency step synchronizes the backend's current `poetry.lock`, including
immutable Git dependencies, and builds the local `autogpt_libs` source. It
reinstalls packages as copies so an earlier pruned runtime, an old package
version, or a shared cache inode cannot survive the refresh. The frontend uses
`pnpm install --prod=false --frozen-lockfile` and regenerates its API before
compiling. Build tools remain installed even with `NODE_ENV=production`.
Retain the complete `verify` and `seal` gates when rebuilding.

`AUTOGPT_DESKTOP_BUILD_HEAP_MB` controls Next.js's heap limit and defaults to
4096 MiB. This runs the frontend build's asset-copy and Next.js steps directly,
because its package script otherwise overrides the heap limit with 16384 MiB.
The runtime contract test checks that the mirrored build script still matches.

The macOS arm64 Erlang artifact requires macOS 15 or later. RabbitMQ 4.3.6
matches the appliance. Erlang 27.3.4.18 remains supported by the
[RabbitMQ 4.3 compatibility matrix](https://www.rabbitmq.com/docs/4.3/which-erlang).
RabbitMQ's Windows and generic Unix hashes come from the
[publisher's release metadata](https://api.github.com/repos/rabbitmq/rabbitmq-server/releases/tags/v4.3.6)
and are enforced by `fetch.py` before extraction.

Existing RabbitMQ 4.1 data requires an intermediate 4.2 upgrade before 4.3,
including enabling stable feature flags, according to the
[publisher's upgrade guide](https://www.rabbitmq.com/docs/4.3/upgrade).
A fresh 4.3 runtime build does not prove this persisted upgrade path. Do not
replace a 4.1 installation with this experimental bundle until that path has
been implemented and validated on an isolated copy of its data.

## CI without a release

`.github/workflows/platform-desktop-build.yml` runs runtime, build, and shell
tests from the PR's merge commit on Windows, macOS, and Linux. It does not call
the fork's release workflow.

Native Windows and macOS packaging is opt-in. A maintainer can add the
`desktop-build` label to a same-repository PR targeting `dev`; remove the label
to stop subsequent pushes from requesting packaging. After the workflow is
available on the repository's default branch, manual dispatch can instead set
`package=true` for a selected ref. Both routes require passing checks first.
The workflow uses an `experimental` variant and a development prerelease
version, supplies no signing secrets, uses a read-only repository token, and
passes `--publish never`. Installers are retained as Actions artifacts for seven
days, named with the tested commit. No GitHub release is created.

These artifacts establish packaging only. Use isolated application data for
native startup, live provider, lifecycle, and persisted-upgrade checks, and
record the build's exact commit and lock hashes with those results. Current
source requires new validation even when the same build machine or downloaded
artifact cache was used before.
