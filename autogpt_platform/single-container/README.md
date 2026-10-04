# AutoGPT Platform

Run the AutoGPT Platform in one container. The image bundles the web app, APIs,
workers, PostgreSQL, RabbitMQ, a three-node Valkey cluster, and FalkorDB-backed
memory while persisting runtime data under `/data`.

> This single-node distribution is experimental. It is intended for local and
> small self-hosted installations, not high-availability deployments.

## Quick start

```bash
docker run -d \
  --name autogpt \
  --restart unless-stopped \
  --shm-size 2g \
  --ulimit nofile=65536:65536 \
  --log-driver json-file \
  --log-opt max-size=50m \
  --log-opt max-file=5 \
  -p 127.0.0.1:3000:3000 \
  -e AUTOGPT_PUBLIC_URL=http://localhost:3000 \
  -v autogpt-data:/data \
  significantgravitas/autogpt:latest
```

The first boot can take several minutes, and it applies the database migrations.
Let it finish: stopping the container during that window can interrupt a
migration, and the next boot will refuse to start until you resolve it (the log
names the migration and the recovery options described in the canonical guide
below). Wait until Docker reports the container as `healthy`, then open
[http://localhost:3000](http://localhost:3000).
Registration starts open; the loopback-only port binding above keeps it local.
If you expose the app to a network, anyone who can reach it can register until
you close signup.

After creating your account, promote it to administrator:

```bash
docker exec autogpt autogpt-admin promote you@example.com
```

Sign out and back in so your session picks up the administrator role.

Then close registration: stop and remove the container
(`docker stop autogpt && docker rm autogpt`) and repeat the quick-start command
with `-e AUTH_ALLOW_NEW_ACCOUNTS=false` added. Keep the same `autogpt-data`
volume so accounts, agents, memory, and generated application secrets survive
container replacement.

## Stopping

The image is designed and tested to stop inside Docker's stock 10-second
timeout. Its measured margin is narrow, and a longer Docker timeout does not
extend Supervisor's internal five-second state-service cap. Larger or slower
state may therefore require normal crash recovery on the next boot.

Agent runs that are still executing are abandoned and may remain displayed as
`RUNNING`. They are not resumed on the next boot, so start a new run.

Supervisor's process names are group-qualified inside the container
(`runtime:rest`, `state:postgres`, and so on). List them with the appliance's
own Supervisor configuration rather than assuming a bare name:

```bash
docker exec autogpt supervisorctl \
  -c /opt/autogpt/single-container/supervisor/supervisord.conf status
```

The one-shot `runtime:bootstrap` shows `EXITED` after a successful start.

## Configuration

`AUTOGPT_PUBLIC_URL` must exactly match the URL used in the browser. For
example, publishing host port `8080` requires
`AUTOGPT_PUBLIC_URL=http://localhost:8080` and `-p 127.0.0.1:8080:3000`.

For LAN or remote access, keep the loopback binding, put a TLS reverse proxy in
front of it, and set `AUTOGPT_PUBLIC_URL` to the proxy's `https://` origin.
Close signup before you expose the app.

Configure model providers and optional integrations with environment variables.
AutoPilot's default remote profile uses `OPEN_ROUTER_API_KEY` for chat and
memory extraction and `OPENAI_API_KEY` for embeddings; the canonical guide
below also covers direct Anthropic and local-model profiles. The
[environment template](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/single-container/.env.example)
lists common settings for `--env-file`. The FalkorDB service always runs, while
Graphiti memory is enabled by the image's default feature configuration; memory
extraction also needs a configured chat model and embedding provider.

The image supports `linux/amd64` and `linux/arm64`. Test installations used
about 5–6 GiB of memory during startup and steady-state health checks, though
actual usage depends on enabled services and workloads. On Docker Desktop, give
its VM more memory than that (Settings → Resources, or `.wslconfig` with the
Windows WSL 2 backend).

The quick-start command's `--shm-size 2g` raises the container's `/dev/shm`
above Docker's 64 MB default; the bundled PostgreSQL and browser tooling both
use shared memory.

## Tags

- `latest` — most recent fully verified AutoGPT Platform release.
- `vX.Y.Z` — immutable image for GitHub release `autogpt-platform-beta-vX.Y.Z`.
- `sha-<git-sha>` — immutable build for the exact source revision of a release.

Legacy `canary-sha-*` tags are unsupported pre-release validation artifacts and
are no longer published.

## Upgrading

Pin a `vX.Y.Z` tag instead of `latest` to choose when you upgrade, and read that
release's notes on the
[releases page](https://github.com/Significant-Gravitas/AutoGPT/releases)
first.

Follow the canonical guide's upgrade steps: pull the new image while the old one
runs, take a cold backup, remove the stopped container, and repeat the
quick-start command with the new tag and the same `autogpt-data` volume. The
first boot of a new version applies database migrations, so let it reach
`healthy` before stopping it. To roll back, restore the pre-upgrade backup into
a new volume and run the previous image. Never start an older image on a volume
that a newer one has migrated.

## More information

- [Canonical single-container operations guide](https://docs.agpt.co/platform/self-hosting/single-container)
- [Release notes](https://github.com/Significant-Gravitas/AutoGPT/releases)
- [AutoGPT repository](https://github.com/Significant-Gravitas/AutoGPT)
- [Security policy](https://github.com/Significant-Gravitas/AutoGPT/security/policy)
- [License](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/LICENSE.md)
