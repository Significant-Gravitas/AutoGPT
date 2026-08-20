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
  -p 127.0.0.1:3000:3000 \
  -e AUTOGPT_PUBLIC_URL=http://localhost:3000 \
  -v autogpt-data:/data \
  significantgravitas/autogpt:latest
```

The first boot can take several minutes, and it applies the database migrations.
Let it finish: stopping the container during that window can interrupt a
migration, and the next boot will refuse to start until you resolve it (the log
names the migration and the command). Wait until Docker reports the container
as `healthy`, then open [http://localhost:3000](http://localhost:3000).
Registration starts open; the loopback-only port binding above keeps it local.
If you expose the app to a network, anyone who can reach it can register until
you close signup.

After creating your account, promote it to administrator:

```bash
docker exec autogpt autogpt-admin promote you@example.com
```

Then recreate the container with `AUTH_ALLOW_NEW_ACCOUNTS=false` to close
registration. Keep the same `autogpt-data` volume so accounts, agents, memory,
and generated application secrets survive container replacement.

## Stopping

`docker stop` completes in well under Docker's stock 10-second timeout, so the
appliance shuts down cleanly on a default host and needs no change to Docker's
host-wide stop timeout. The bundled PostgreSQL, RabbitMQ, Valkey and FalkorDB
are drained last and in full; everything stateless is stopped first, and is
terminated rather than waited on once it has released what it holds.

Agent runs that are still executing do not survive the stop. They are not
resumed on the next boot, so re-run them.

Supervisor's process names are group-qualified inside the container -- use
`supervisorctl status` to see them (`runtime:rest`, `state:postgres`, and so
on) rather than assuming a bare name.

## Configuration

`AUTOGPT_PUBLIC_URL` must exactly match the URL used in the browser. For
example, publishing host port `8080` requires
`AUTOGPT_PUBLIC_URL=http://localhost:8080` and `-p 127.0.0.1:8080:3000`.

Configure model providers and optional integrations with environment variables.
Core memory is enabled and stored in the bundled FalkorDB; memory extraction
also needs a configured chat model and embedding provider.

The image supports `linux/amd64` and `linux/arm64`. Test installations used
about 5–6 GiB of memory during startup and steady-state health checks, though
actual usage depends on enabled services and workloads.

ChatGPT/Codex temporary auth homes use `/dev/shm/autogpt-codex` automatically.
The quick-start command's `--shm-size 2g` keeps that credential material in
memory rather than the container's writable layer.

## Tags

- `latest` — most recent fully verified AutoGPT Platform release.
- `vX.Y.Z` — immutable AutoGPT Platform release.
- `sha-<git-sha>` — immutable build for an exact source revision.

## More information

- [AutoGPT repository](https://github.com/Significant-Gravitas/AutoGPT)
- [Security policy](https://github.com/Significant-Gravitas/AutoGPT/security/policy)
- [License](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/LICENSE.md)
