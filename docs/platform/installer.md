# AutoGPT Platform Release Installer

The release installer runs the published AutoGPT Platform appliance as one
Docker container. It does not clone the repository, build AutoGPT, install
Docker, elevate privileges, or accept Docker Desktop license terms.

The appliance exposes one loopback-only port. Application state and generated
secrets live in a named Docker volume; installer identity and runtime
configuration live in a dedicated private state directory.

## Prerequisites

Before running the installer:

- Install and start Docker from the
  [official Docker Engine](https://docs.docker.com/engine/install/) or
  [Docker Desktop](https://docs.docker.com/desktop/) documentation.
- Select a local `unix://` Docker endpoint on Linux or macOS. Remote and
  non-Unix contexts are rejected.
- Configure the daemon for Linux containers on `amd64` or `arm64`.
- Allow about 25 GB of free disk; at least 8 GB RAM is recommended.

Docker Desktop is a separate product with its own license terms. AutoGPT does
not install it or grant a Docker license.

## Install (After the Release Gates Pass)

> [!WARNING]
> Do not use the hosted command yet. `setup.agpt.co/install.sh` still serves
> the legacy Compose installer, and the appliance image tags are not public.
> Maintainers must complete every
> [release gate](#maintainer-release-gates) before exposing these commands in
> the README or getting-started guide.

This first appliance-installer release supports Linux and macOS. Windows is
intentionally withheld until its standard-user filesystem checks and native
Docker argument handling are validated; use the manual setup guide there.

The command below uses a unique temporary file, executes it only after the HTTPS
download succeeds, and removes it afterward. Do not stream a network response
directly into a shell.

### Linux and macOS

```bash
(
  installer="$(mktemp)" &&
  trap 'rm -f "$installer"' EXIT &&
  curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL \
    -o "$installer" https://setup.agpt.co/install.sh &&
  bash "$installer"
)
```

### Options

| Goal | Linux/macOS |
| --- | --- |
| Current published appliance | _(no flag)_ |
| Specific published version | `--release=vX.Y.Z` |
| Appliance release-tag form | `--release=autogpt-platform-beta-vX.Y.Z` |
| Hardware and Docker checks only | `--preflight-only` |
| Print image selection only | `--resolve-only` |
| Skip RAM/disk checks | `--skip-preflight` |
| Custom private state directory | `--dir=PATH` |

`--skip-preflight` does not skip the local-endpoint, Linux-container, or
architecture checks.

The default state directory is `$XDG_CONFIG_HOME/autogpt`, or
`$HOME/.config/autogpt` when `XDG_CONFIG_HOME` is unset. A custom directory
must be user-owned and private; the installer rejects symlink paths and shared
or unsafe locations.

## Artifact and Runtime Contract

With no release flag, the installer pulls
`significantgravitas/autogpt:latest`. An explicit `vX.Y.Z` selects the matching
version tag. After pulling, the installer:

1. Requires the expected OCI title and source repository plus a well-formed
   40-hex source revision label.
2. Requires a Linux image matching the local daemon's `amd64` or `arm64`
   architecture.
3. Resolves exactly one native `RepoDigest` and runs that immutable digest,
   never the mutable tag.
4. Creates an installer-owned, labelled `autogpt-platform-data` volume.
5. Starts `autogpt` with `127.0.0.1:3000`, the private environment file,
   restart and stop policies, log rotation, shared memory, and file-descriptor
   limits.

The bootstrap authenticates transport with HTTPS, and the installer pins the
pulled image by digest after checking its appliance metadata. It does not
independently verify a publisher signature or protect against compromise of
the image-publishing credentials. Signed release artifacts are a follow-up,
not a claim made by this installer.

The installer refuses to adopt an unlabelled container or volume. A rerun is a
no-op, or starts a stopped container, only when the installer identity,
environment hash, immutable image digest, process, port, mount, privilege,
restart, logging, memory, and limit settings all match. Drift fails closed.

If established installer lifecycle state exists but its named volume is
missing, the installer refuses to create an empty replacement. An interrupted
first pull has no established marker and can be retried safely. Restore a
previously established volume or choose a new state directory.

## Runtime Configuration

The private `autogpt.env` initially contains:

```dotenv
AUTOGPT_PUBLIC_URL=http://localhost:3000
```

Add provider keys and other appliance variables there. The installer binds the
environment file's SHA-256 hash to the container label. To apply a deliberate
configuration change, stop and remove only the container, then rerun the same
installer release:

```bash
docker stop --time 360 autogpt
docker rm autogpt
```

Do not remove `autogpt-platform-data`; it contains accounts, agents, generated
secrets, and application state.

## First Run

After the installer reports a healthy container, open
[http://localhost:3000](http://localhost:3000). Registration starts open so
you can create the intended account. Promote it:

```bash
docker exec autogpt autogpt-admin promote you@example.com
```

Then set this in `autogpt.env`, recreate the container as described above, and
verify registration is closed:

```dotenv
AUTH_ALLOW_NEW_ACCOUNTS=false
```

Keep the default loopback binding. To serve other machines, place AutoGPT
behind a TLS reverse proxy and set `AUTOGPT_PUBLIC_URL` to the exact public URL.

## Upgrades

Before an upgrade, back up `autogpt-platform-data`. Stop and remove only the
`autogpt` container, then rerun the installer with the intended release. The
installer will reuse the named volume only when its ownership labels and the
private installer identity still match.

## Maintainer Release Gates

Repository CI cannot prove the public bootstrap handoff. Do not publish or
announce the installation commands until every external gate below passes:

- Deploy the new repository `install.sh` through the Caddy or object-storage
  configuration behind `setup.agpt.co`. The endpoint must no longer serve the
  legacy clone/Compose installer.
- Publish `significantgravitas/autogpt:vX.Y.Z` and `:latest` only after the
  multi-architecture workflow smoke-tests and scans both runnable images.
- Ensure the appliance release's tag commit contains the full installer,
  publication workflow, and helper, then verify the release-triggered run.
  Manual development dispatches publish SHA artifacts, not the release channel
  used by this installer.
- Verify both public tags expose Linux `amd64` and `arm64` manifests and the
  expected OCI identity:

  ```bash
  docker buildx imagetools inspect significantgravitas/autogpt:vX.Y.Z
  docker buildx imagetools inspect significantgravitas/autogpt:latest
  ```

- Fetch the hosted installer into a new temporary file from an external
  machine and compare it with the released repository file. On each supported
  operating system, perform a clean install against the public image, wait for
  health, verify the loopback endpoint, rerun to prove exact-contract
  idempotency, and confirm no mutable tag was used to create the container.

## Legacy setup scripts

For a prebuilt, experimental alternative intended for small self-hosted
installations, see [Run AutoGPT in one Docker container](single-container.md).
That image is separate from the development-oriented installer described here.

### Manual Installation

If you prefer, you can manually download and run the installer scripts:

- **Linux/macOS:** `setup-autogpt.sh`
- **Windows:** `setup-autogpt.bat`

These scripts are located in the `autogpt_platform/installer/` directory.

### Running fully offline with a local LLM (Ollama)

Both installer scripts accept an opt-in flag that installs
[Ollama](https://ollama.com), pulls a default chat model, and wires
`backend/.env` so AutoPilot runs **without any cloud API keys**. This
is useful for air-gapped or privacy-sensitive deployments — see
[Running AutoPilot on a self-hosted LLM](copilot-local-llm.md) for the
full reference.

#### Linux / macOS

```bash
cd autogpt_platform/installer
./setup-autogpt.sh --with-ollama
# Optional overrides:
#   --ollama-model=qwen3:14b-instruct-q4_K_M
#   --ollama-host=http://gpu-rig.lab:11434   # use an existing Ollama
```

#### Windows

```cmd
cd autogpt_platform\installer
setup-autogpt.bat /with-ollama
REM Optional overrides:
REM   /ollama-model=qwen3:14b-instruct-q4_K_M
REM   /ollama-host=http://gpu-rig.lab:11434
```

The installer:

1. Installs Ollama (skipped if already present, or if `--ollama-host` points at an existing one).
2. Configures `OLLAMA_HOST=0.0.0.0:11434` + `OLLAMA_CONTEXT_LENGTH=32768` so containers can reach it and so AutoPilot's ~8 k system prompt isn't truncated by Ollama's 4 k default.
3. Pulls the chat model (default `hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M`).
4. Appends a marker-bounded block to `autogpt_platform/backend/.env` with `CHAT_USE_LOCAL=true` plus the `CHAT_BASE_URL` / `CHAT_API_KEY` / `CHAT_*_MODEL` overrides.

Re-running with `--with-ollama` is idempotent — the wiring block is rewritten in place.

## Troubleshooting

1. Confirm `docker info` succeeds against a local Linux-container daemon.
2. If an image pull fails just after release, wait for public manifest
   publication and retry; the installer never falls back to a source build.
3. If the installer reports state or runtime drift, preserve the named volume,
   inspect the existing container, and follow the explicit upgrade procedure.
4. Check `docker logs --tail 100 autogpt` when health checks fail.
