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
  --stop-timeout 360 \
  --shm-size 2g \
  --ulimit nofile=65536:65536 \
  -p 127.0.0.1:3000:3000 \
  -e AUTOGPT_PUBLIC_URL=http://localhost:3000 \
  -v autogpt-data:/data \
  significantgravitas/autogpt:latest
```

The first boot can take several minutes. Wait until Docker reports the container
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
