# Run AutoGPT in one Docker container (Experimental)

!!! danger
    This distribution is experimental. It is intended for local evaluation and
    single-host self-hosting, not production. Do not publish it to Docker Hub
    until the license, image-content, vulnerability, fresh-install, upgrade, and
    rollback gates in this document have been completed.

The single-container image packages the AutoGPT Platform web application and
all of its supporting processes behind one public port. A process supervisor
inside the container manages the frontend, backend roles, PostgreSQL, a
three-node Valkey cluster, RabbitMQ, ClamAV, and FalkorDB. You do not need to
install or connect a separate Supabase stack.

This is a distribution format, not a replacement for the existing
multi-container development Compose stack.

This page covers:

- [quick start](#quick-start), including a custom host port;
- [provider API keys and local or remote LLMs](#llm-providers-and-local-inference);
- [accounts, external API access, OAuth, and email](#accounts-and-the-first-administrator);
- [the `/data` volume, backup, restore, and upgrades](#persistence-and-secret-custody);
- [common problems](#troubleshooting); and
- [the extra controls required before publishing an image](#docker-hub-publication-controls).

## Quick start

An image tagged `autogpt` has a complete default command. The literal command
below starts the whole appliance in the foreground:

```bash
docker run autogpt
```

That form is useful as a boot check, but it does not publish the web port and
Docker gives `/data` an anonymous volume. For an installation you can open,
restart, and upgrade, first create a private configuration file:

```bash
install -m 600 autogpt_platform/single-container/.env.example \
  autogpt_platform/single-container/.env
```

Then run the image with a published port and named data volume:

```bash
docker run --detach --name autogpt \
  --restart unless-stopped \
  --stop-timeout 360 \
  --shm-size 2g \
  --ulimit nofile=65536:65536 \
  --log-driver json-file \
  --log-opt max-size=50m \
  --log-opt max-file=5 \
  --add-host host.docker.internal:host-gateway \
  --env-file autogpt_platform/single-container/.env \
  -p 127.0.0.1:3000:3000 \
  -v autogpt-platform-data:/data \
  autogpt
```

Open `http://localhost:3000`. First startup can take several minutes while the
volume is initialized, migrations run, and bundled services become ready.
Check the full appliance health and follow its logs with:

```bash
docker inspect --format '{{.State.Health.Status}}' autogpt
docker logs --follow autogpt
```

Pressing `Ctrl-C` while following logs does not stop the detached container.
Stop and start it without losing the named volume with:

```bash
docker stop --time 360 autogpt
docker start autogpt
```

To use host port `3202`, keep the container-side port at `3000`, change the
mapping to `-p 127.0.0.1:3202:3000`, and set this in the environment file:

```dotenv
AUTOGPT_PUBLIC_URL=http://localhost:3202
```

The public URL must match the exact origin in the browser. Docker cannot infer
the host-facing port from inside the container; a mismatch breaks auth actions,
callbacks, cookies, and generated links.

## Host requirements

Use a current Docker Engine or Docker Desktop release with the Docker Compose
plugin. The image targets `linux/amd64` and `linux/arm64`.

The following are conservative planning values, not measured release
requirements:

- 4 CPU cores and 12 GiB of RAM are recommended.
- 8 GiB of RAM may be enough for light evaluation with FalkorDB and ClamAV
  disabled, but must be validated against the intended workload.
- Reserve at least 25 GiB of free disk for the image, databases, ClamAV
  signatures, browser artifacts, logs, an upgrade, and one local backup.
- Compose allocates 2 GiB of shared memory by default for Chromium. Override it
  with `AUTOGPT_SHM_SIZE` only after browser-tool testing.

The image also caps each Python role's Prisma pool at five connections so the
bundled roles do not scale their pools with the host CPU count and exhaust the
single PostgreSQL server. Treat `DB_CONNECTION_LIMIT` as a whole-appliance
capacity setting: increasing it requires accounting for every backend role and
the frontend auth pool, not just one process.

Before publishing any tag, record the compressed image size, unpacked size,
idle and peak memory, startup time, and steady-state disk growth on both target
architectures. The image intentionally includes several substantial services,
so it should not be presented as a small image.

## Build and start from source

From the repository root:

```bash
install -m 600 autogpt_platform/single-container/.env.example \
  autogpt_platform/single-container/.env

docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  up --build --detach --wait --wait-timeout 900
```

`--wait` returns only after Docker reports the complete appliance healthy. If
startup fails or times out, inspect it with:

```bash
docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  logs -f autogpt
```

To build the standalone image as `autogpt` without starting Compose:

```bash
docker build \
  --target single-container \
  -f autogpt_platform/backend/Dockerfile \
  -t autogpt .
```

Run that image with the canonical command in [Quick start](#quick-start).

The internal watchdog deliberately stops an appliance that remains unhealthy.
Use `always` or `unless-stopped`, as shown above, if Docker should recover it;
`on-failure` is not sufficient because Supervisor can complete a coordinated
shutdown with exit code zero. Compose uses `unless-stopped`.

Do not copy any of the development `.env.default` files into this image's
configuration. They contain public development credentials. The entrypoint
generates private internal database, queue, encryption, auth, and signing
secrets on first boot and stores them under `/data`; subsequent boots reuse
them.

## Run a published experimental image

Published tags must be explicit and immutable in practice. `latest` is not
created by the release workflow.

Edit `autogpt_platform/single-container/.env`:

```dotenv
AUTOGPT_IMAGE=YOUR_DOCKERHUB_ACCOUNT/autogpt-platform
AUTOGPT_TAG=experimental-YYYYMMDD-SHORT_SHA
```

Then pull and start without building local source:

```bash
docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  pull autogpt

docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  up --detach --no-build --wait --wait-timeout 900
```

For direct Docker, pull the same immutable image reference and use it in place
of the final `autogpt` argument in the [Quick start](#quick-start) command:

```bash
docker pull YOUR_DOCKERHUB_ACCOUNT/autogpt-platform:experimental-YYYYMMDD-SHORT_SHA
```

Record the registry manifest digest from
`docker buildx imagetools inspect IMAGE:TAG` in the deployment record. A local
image ID from `docker compose images` is not the multi-platform registry
manifest digest, and a mutable tag alone is not a sufficient rollback record.

## Configure the container

Keep installation settings in
`autogpt_platform/single-container/.env`, set its mode to `0600`, and pass it to
every Compose command with `--env-file`. Direct `docker run` users pass the same
file with Docker's `--env-file` flag. Do not copy a development `.env.default`;
those files contain public development credentials.

Some values configure Docker Compose itself, while others are read inside the
container:

| Purpose | Docker Compose | Direct `docker run` |
| --- | --- | --- |
| Image | `AUTOGPT_IMAGE` and `AUTOGPT_TAG` | final image argument |
| Host bind and port | `AUTOGPT_BIND_ADDRESS` and `AUTOGPT_PORT` | `-p HOST_IP:HOST_PORT:3000` |
| Persistent volume | `AUTOGPT_DATA_VOLUME` | `-v VOLUME_NAME:/data` |
| Shared memory | `AUTOGPT_SHM_SIZE` | `--shm-size` |
| Browser-visible origin | `AUTOGPT_PUBLIC_URL` | `AUTOGPT_PUBLIC_URL` in `--env-file` |

Changing `AUTOGPT_PORT` inside a direct-run container does nothing; direct
users must change `-p`. With Compose, change `AUTOGPT_PORT` and
`AUTOGPT_PUBLIC_URL` together. The internal web port always remains `3000`.

Most in-container setting changes require recreating the container. Compose
does that safely while retaining the named volume:

```bash
docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  up --detach --no-build --force-recreate --wait --wait-timeout 900
```

For direct Docker, stop and remove only the container, then repeat the Quick
start command with the same named volume and updated environment file:

```bash
docker stop --time 360 autogpt
docker rm autogpt
```

Removing the container does not remove `autogpt-platform-data`. Do not add
`--volumes`, delete the named volume, or change its name unless you intend to
start a different installation.

## Public URL, port, and TLS

`AUTOGPT_PUBLIC_URL` must be the exact origin users visit, including `https://`
and any non-default port, with no path component. Set it correctly before the
first signup because it is used for auth issuer, cookies, OAuth callbacks, and
links sent by email.

For a remote host:

```dotenv
AUTOGPT_PUBLIC_URL=https://agents.example.com
AUTOGPT_BIND_ADDRESS=127.0.0.1
AUTOGPT_PORT=3000
```

The default bind address is loopback. Put a TLS reverse proxy on the same host
in front of `127.0.0.1:3000`; do not expose the port directly to the internet
over plain HTTP. `AUTOGPT_PUBLIC_URL` is authoritative; the appliance validates
it and derives its upstream host and protocol from it instead of trusting
inbound forwarded-origin headers. The outer proxy must:

- terminate TLS and route every path to port `3000`;
- support WebSocket upgrades for `/_agpt/ws`;
- disable response buffering and allow long timeouts for streaming requests
  below `/_agpt/api`; and
- forward all other paths, including `/api/auth/*`, to port `3000` unchanged.

Set `AUTOGPT_BIND_ADDRESS=0.0.0.0` only when direct LAN access is intentional
and the network itself supplies the required protection. Internal PostgreSQL,
Redis, RabbitMQ, ClamAV, FalkorDB, and backend ports are not published.

## Accounts and the first administrator

New installations use open email/password signup by default. To restrict
signup before exposing the host, set either:

```dotenv
AUTH_ALLOW_NEW_ACCOUNTS=false
```

or a comma-separated exact-email/domain allowlist:

```dotenv
AUTH_SIGNUP_ALLOWLIST=owner@example.com,@example.org
```

Create the intended account through the UI, then promote it from the host:

```bash
docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  exec autogpt autogpt-admin promote user@example.com
```

For a direct-Docker installation:

```bash
docker exec autogpt autogpt-admin promote user@example.com
```

The account must already exist and the email must identify exactly one user.
After promotion, sign out and sign in again so the session receives the new
role.

Remove any temporary signup allowance after the account exists. Keep host-level
access to `docker compose exec` limited to trusted administrators.

## Create an AutoGPT API key

Provider keys let AutoGPT call model vendors. An **AutoGPT API key** is
different: it lets another application call this AutoGPT installation.

1. Sign in and open **Settings > AutoGPT API Keys**, or visit
   `/settings/api-keys` on your installation.
2. Create a key and copy it immediately.
3. Send it in the `X-API-Key` header.

For example, a local installation can list available blocks with:

```bash
curl --fail --show-error \
  -H 'X-API-Key: YOUR_AUTOGPT_API_KEY' \
  http://localhost:3000/_agpt/external-api/v1/blocks
```

Replace the origin if you use another port or hostname. The bundled image does
not expose internal OpenAPI, metrics, or backend documentation endpoints. See
the [API integration guide](integrating/api-guide.md) for authentication and
API concepts; substitute your self-hosted origin for its hosted examples.

## LLM providers and local inference

The image starts without a model-provider key. AutoPilot needs one explicit
routing profile; put one of the following profiles in the installation `.env`,
then recreate the container as described in
[Configure the container](#configure-the-container).

### OpenRouter (recommended remote profile)

The default AutoPilot route is OpenRouter:

```dotenv
CHAT_USE_LOCAL=false
OPEN_ROUTER_API_KEY=YOUR_OPENROUTER_KEY
```

Use `OPEN_ROUTER_API_KEY` for this route. Do not set only a genuine OpenAI key:
the compatibility fallback can otherwise send that key to the OpenRouter
endpoint and receive `401 Unauthorized`.

### Direct Anthropic

To bypass OpenRouter and call Anthropic directly:

```dotenv
CHAT_USE_LOCAL=false
CHAT_USE_OPENROUTER=false
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY
```

This is an advanced profile; any explicit model overrides must use models that
the direct Anthropic API accepts.

### Ollama on the Docker host

Run an OpenAI-compatible model with tool/function-calling and streaming support,
then configure AutoPilot like this:

```dotenv
CHAT_USE_LOCAL=true
CHAT_BASE_URL=http://host.docker.internal:11434/v1
CHAT_API_KEY=ollama
CHAT_FAST_STANDARD_MODEL=hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M
```

`CHAT_API_KEY` must be non-empty even if the local server ignores it, and
`CHAT_FAST_STANDARD_MODEL` must exactly match a model already installed on the
inference server. For Ollama, use a context window of at least 32,768 tokens and
make sure it listens on an address Docker can reach. A direct run needs the
`--add-host host.docker.internal:host-gateway` flag from Quick start; Compose
adds that host mapping automatically.

Test host connectivity from a direct-run container with:

```bash
docker exec autogpt \
  curl --fail --show-error http://host.docker.internal:11434/api/tags
```

For Compose, replace `docker exec autogpt` with the documented
`docker compose ... exec autogpt` command. See the
[local LLM guide](copilot-local-llm.md) for model sizing, Ollama host setup,
context length, advanced tiers, and security guidance. The
[Ollama blocks guide](ollama.md) covers Ollama credentials used by agent blocks;
those are separate from AutoPilot's `CHAT_*` settings.

### Remote OpenAI-compatible endpoint

The same custom transport works with a remote vLLM, LocalAI, LM Studio,
LiteLLM, or another OpenAI-compatible HTTPS endpoint:

```dotenv
CHAT_USE_LOCAL=true
CHAT_BASE_URL=https://llm.example.com/v1
CHAT_API_KEY=YOUR_ENDPOINT_KEY
CHAT_FAST_STANDARD_MODEL=YOUR_SERVER_MODEL_ID
```

Here `CHAT_USE_LOCAL` selects the custom OpenAI-compatible transport; the server
does not have to run on the same machine. Do not expose an unauthenticated local
model server to the internet.

### Other provider and block keys

These installation-wide variables support other model-backed features:

| Variable | Used for |
| --- | --- |
| `OPEN_ROUTER_API_KEY` | Default AutoPilot remote route |
| `ANTHROPIC_API_KEY` | Direct Anthropic AutoPilot profile and compatible blocks |
| `OPENAI_API_KEY` | OpenAI-backed agent blocks and default transcription; it does not select direct OpenAI AutoPilot routing |
| `GROQ_API_KEY` | Groq-backed agent blocks |

An alternative transcription service can be configured with
`TRANSCRIPTION_API_BASE_URL`, `TRANSCRIPTION_API_KEY`, and
`TRANSCRIPTION_MODEL`. Additional provider and block variables placed in the
same `.env` are passed to the application processes even when they are not
listed in `.env.example`.

Users can also add per-user block credentials under **Settings > Integrations**.
Those encrypted credentials are stored in `/data`; installation-wide values in
the host `.env` remain outside the volume.

Keep the host `.env` at mode `0600`. Environment credentials are visible to
Docker-daemon administrators and through privileged container inspection. The
appliance-generated database, encryption, and session secrets are stored in
`/data/config/runtime.env` at mode `0600`, but anyone with Docker or volume
access can still read them.

## Self-hosted telemetry

The single-container image does not send product analytics by default, even
after a user accepts the analytics cookie category. An operator can explicitly
opt in to the AutoGPT project's Google Analytics property, or provide a
different property:

```dotenv
AUTOGPT_TELEMETRY_ENABLED=true
AUTOGPT_GA_MEASUREMENT_ID=G-EXAMPLE
```

Consent in the browser remains required after the operator enables telemetry.
Leave `AUTOGPT_TELEMETRY_ENABLED=false` for a telemetry-free installation.
The external Tally feedback widget is separately disabled because loading it
contacts Tally before a form is opened. Set `AUTOGPT_FEEDBACK_ENABLED=true`
only if that integration is wanted. Developer overlays are also off by default;
`AUTOGPT_DEVELOPER_UI_ENABLED=true` enables the Agentation overlay, while React
Query devtools require a custom frontend build as well.

## OAuth and email

Social login and agent integration OAuth are separate credential sets.

Social-login providers use `AUTH_*` variables:

```dotenv
AUTH_GOOGLE_CLIENT_ID=
AUTH_GOOGLE_CLIENT_SECRET=
AUTH_GITHUB_CLIENT_ID=
AUTH_GITHUB_CLIENT_SECRET=
AUTH_DISCORD_CLIENT_ID=
AUTH_DISCORD_CLIENT_SECRET=
```

Agent block integrations use their unprefixed equivalents, such as
`GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`. Google Drive/Picker integrations
also use `GOOGLE_API_KEY` and the numeric Cloud project ID in `GOOGLE_APP_ID`.
Their shared callback is:

```text
https://agents.example.com/auth/integrations/oauth_callback
```

Social login providers use these Better Auth callbacks:

```text
https://agents.example.com/api/auth/callback/google
https://agents.example.com/api/auth/callback/github
https://agents.example.com/api/auth/callback/discord
```

Create provider applications only after `AUTOGPT_PUBLIC_URL` and TLS are final.
Replace the example origin and copy callback URLs exactly; login credentials and
agent-integration credentials are not interchangeable even for the same vendor.

Configure Postmark before relying on password reset, verified email changes, or
notification email:

```dotenv
POSTMARK_SERVER_API_TOKEN=
POSTMARK_SENDER_EMAIL=autogpt@example.com
POSTMARK_WEBHOOK_TOKEN=
```

For remote deployments, also replace the default web-push contact with an
address monitored by the operator:

```dotenv
VAPID_CLAIM_EMAIL=mailto:admin@example.com
```

Keep `AUTH_REQUIRE_EMAIL_VERIFICATION=false` for this experimental image. The
current signup/provisioning path assumes an immediate session; enabling required
verification can leave a newly verified auth identity without its platform user
provisioning. Postmark configuration does not remove that limitation.

## Optional bundled processes

The default enables the complete local stack:

```dotenv
AUTOGPT_ENABLE_FALKORDB=true
AUTOGPT_ENABLE_CLAMAV=true
AUTOGPT_ENABLE_BOT_SERVICES=false
```

Disable FalkorDB or ClamAV only after verifying which product features become
unavailable. Bot services should remain off unless their platform tokens and
public webhook routes are configured. These switches stop processes; they do
not remove their binaries from an already-built image.

## Authentication and the old Supabase stack

A fresh install has no Supabase runtime dependency. Better Auth runs in the
frontend process and uses the bundled PostgreSQL database. The database still
creates a small `auth.users` compatibility shim because historical Prisma
migrations refer to that schema; the shim is not a GoTrue server and does not
make Supabase a runtime dependency.

Legacy shared-secret JWT verification is intentionally off:

```dotenv
AUTOGPT_ENABLE_LEGACY_AUTH=false
```

Only an operator migrating active sessions from an older Supabase-based install
should opt in. Use the old installation's real secret, never the public
development value from a checked-in `.env.default`, and set a short bridge
window:

```dotenv
AUTOGPT_ENABLE_LEGACY_AUTH=true
SUPABASE_JWT_SECRET=REDACTED_OLD_INSTALLATION_SECRET
SUPABASE_BRIDGE_MAX_TOKEN_AGE_DAYS=30
```

Treat this as a temporary migration state. A leaked HS256 secret can be used to
forge privileged tokens. Turn the bridge off and remove the secret as soon as
the migration window closes. Fresh installs must leave both legacy secret
variables unset.

## Persistence and secret custody

All durable application state lives under `/data`. Compose mounts the named
volume `autogpt-platform-data` there by default; its name is configurable with
`AUTOGPT_DATA_VOLUME`. The recommended direct command mounts the same named
volume with `-v autogpt-platform-data:/data`.

A literal `docker run autogpt` instead gets an anonymous volume. Recreating the
container can strand that state, and `docker run --rm autogpt` removes its
anonymous volume when the container exits. Always name the volume for anything
other than a throwaway boot test. To see which volume a running direct container
uses:

```bash
docker inspect --format \
  '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}' \
  autogpt
```

The volume includes:

| Path below `/data` | Contents |
| --- | --- |
| `config` | Generated runtime secrets and backend configuration |
| `postgres` | PostgreSQL databases, including auth and platform data |
| `rabbitmq` and `valkey` | Queue and three-node Valkey state |
| `falkordb` and `clamav` | Graph memory and antivirus signatures/state |
| `workspaces` and `home` | User workspaces and application home data |
| `cache` | Persistent application caches, including the Next.js cache |

The host `.env`, TLS/reverse-proxy configuration, Docker logs, selected image
reference and registry digest are **not** stored in `/data`. Retain them with
the backup record.

Consequences:

- `docker compose down` preserves data; `docker compose down -v` deletes it.
- Losing the generated secret file can make encrypted credentials unusable and
  invalidate every session, even if a database copy survives.
- A volume backup contains user content, provider credentials, auth keys, and
  database passwords. Encrypt it at rest and tightly restrict access.
- Do not mount one volume into two running AutoGPT containers.
- The Compose volume name is host-global. Give a second installation a unique
  `AUTOGPT_DATA_VOLUME`; never use one volume concurrently for two installs.

### Cold backup

A stopped-volume archive is the supported first backup mechanism for this
experimental image. It is intentionally simple and restores all coupled
services to the same point in time.

```bash
umask 077
export AUTOGPT_IMAGE_REF=autogpt
# For a published image, use:
# export AUTOGPT_IMAGE_REF=YOUR_DOCKERHUB_ACCOUNT/autogpt-platform:experimental-YYYYMMDD-SHORT_SHA
export AUTOGPT_DATA_VOLUME=autogpt-platform-data

docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  down

install -m 600 /dev/null autogpt-platform-data.tgz
docker run --rm \
  --entrypoint tar \
  -v "${AUTOGPT_DATA_VOLUME}:/data:ro" \
  -v "${PWD}:/backup" \
  "${AUTOGPT_IMAGE_REF}" \
  -czf /backup/autogpt-platform-data.tgz -C /data .
```

Keep the image tag, manifest digest, Git commit, configuration file, and backup
checksum beside the archive. The archive is plaintext and contains the complete
secret-bearing appliance state; encrypt it immediately with an approved backup
mechanism, restrict the encrypted destination, and securely remove any staging
copy. Test restoration regularly; an untested archive is not a backup plan.

### Restore without overwriting the old volume

Restore into a new named volume so the source remains recoverable:

```bash
export RESTORED_VOLUME=autogpt-platform-data-restored-YYYYMMDD
docker volume create "${RESTORED_VOLUME}"

docker run --rm \
  --entrypoint tar \
  -v "${RESTORED_VOLUME}:/data" \
  -v "${PWD}:/backup:ro" \
  "${AUTOGPT_IMAGE_REF}" \
  -xzf /backup/autogpt-platform-data.tgz -C /data

AUTOGPT_DATA_VOLUME="${RESTORED_VOLUME}" \
docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  up --detach --no-build --wait --wait-timeout 900
```

Verify full Docker health, login, saved credentials, and workspace contents
before changing the deployment's normal volume setting. Stop the restored test
afterward and retain the source volume until the restore is accepted. Never
selectively mix databases or service directories from different backups.

## Upgrade and rollback

For an upgrade:

1. Record the running image digest and configuration.
2. Stop the service and take a cold backup.
3. Set a new explicit experimental tag and pull it.
4. Start with `--no-build`; startup applies database migrations before serving
   application traffic.
5. Wait for health, then test login, an agent execution, streaming, WebSockets,
   browser tools, and persistence across one restart.

```bash
docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  pull autogpt

docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  up --detach --no-build --wait --wait-timeout 900
```

For a direct-Docker installation, take the same cold backup, then replace only
the container while retaining its named volume:

```bash
docker pull NEW_IMMUTABLE_IMAGE_REFERENCE
docker stop --time 360 autogpt
docker rm autogpt
```

Repeat the Quick start command with the same `--env-file`, port mapping, and
`-v autogpt-platform-data:/data`, but use
`NEW_IMMUTABLE_IMAGE_REFERENCE` as the final argument.

Do not run an older image against a volume after newer migrations have modified
it. A rollback means selecting the prior image tag **and restoring the matching
pre-upgrade archive into a new volume**. This is why the backup and image digest
are part of the same deployment record.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `docker run autogpt` starts but the browser cannot connect | The literal command does not publish a port. Use `-p 127.0.0.1:3000:3000`, or the complete Quick start command. |
| Port `3202` is unreachable | Publish `-p 127.0.0.1:3202:3000` for direct Docker. With Compose, set `AUTOGPT_PORT=3202`. In both cases set `AUTOGPT_PUBLIC_URL=http://localhost:3202` and recreate the container. |
| Signup or login reports a Server Action, origin, callback, or forwarded-host error | `AUTOGPT_PUBLIC_URL` does not exactly match the browser origin. Fix the scheme, hostname, and port in `.env`, then recreate the container. |
| The container is `starting` or `unhealthy` | First boot can take several minutes. Read `docker logs autogpt`, then run `docker exec autogpt /usr/local/bin/autogpt-healthcheck`. `GET /healthz` checks only nginx; Docker health checks all bundled services and app roles. |
| Provider changes have no effect | Make sure direct Docker includes `--env-file`. For Compose, pass the same file with Compose's `--env-file`, then force-recreate. Use `OPEN_ROUTER_API_KEY` for the default remote route instead of only `OPENAI_API_KEY`. |
| AutoPilot receives `401 Unauthorized` from OpenRouter | Configure a real `OPEN_ROUTER_API_KEY`, or select the direct Anthropic/custom compatible profile explicitly. |
| Ollama cannot be reached | Confirm the server listens beyond host loopback, the container has the `host.docker.internal` mapping, and the in-container `/api/tags` connectivity check succeeds. |
| Ollama returns model-not-found | Set `CHAT_FAST_STANDARD_MODEL` to the exact installed model ID. Pull that model on the Ollama host before recreating AutoGPT. |
| The container shuts down after staying unhealthy | The watchdog intentionally exits the appliance cleanly. Use Docker restart policy `always` or `unless-stopped`; `on-failure` does not restart a clean exit. |
| Data appears missing after recreation | The new container is using another or anonymous `/data` volume. Inspect its mount name and reattach the correct named volume; do not delete either volume while investigating. |

When sharing diagnostics, redact secrets. In particular, unredacted
`docker inspect` and `docker compose config` output can contain every provider,
OAuth, and email credential from the environment file.

## Operational validation

Resolve the Compose model before every release:

```bash
docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  config --quiet

docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  config --services
```

The second command must print only `autogpt`. Be aware that an unredacted
`docker compose config` can print interpolated secrets; do not attach it to an
issue or CI log.

Check container and image health:

```bash
docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  ps

docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  exec autogpt /usr/local/bin/autogpt-healthcheck

curl --fail --show-error http://127.0.0.1:3000/
```

A publication candidate must additionally pass:

- a fresh empty-volume boot on both `linux/amd64` and `linux/arm64`;
- signup, login, logout, password reset, admin promotion, and token refresh;
- backend REST, long-lived streaming, and WebSocket traffic through port 3000;
- one agent execution, scheduled execution, notification, file scan, FalkorDB
  memory operation, and Chromium browser-tool run;
- one real local `bash_exec` sandbox command as the unprivileged application
  user under the target Docker Engine's default seccomp and user-namespace
  settings on each architecture;
- graceful `docker stop`, automatic restart, and persistence across restart;
- cold backup, restore into a new volume, forward upgrade, and rollback drill;
- inspection of `docker history` and image layers for copied `.env` files or
  credentials; and
- a vulnerability scan, SBOM review, and provenance verification for the exact
  pushed digest.

## Docker Hub publication controls

`.github/workflows/platform-single-container-docker.yml` builds, boots through
both literal `docker run IMAGE` and the one-service Compose file, exercises an
automatic watchdog-driven restart, persistence-checks, and scans `linux/amd64`
and `linux/arm64` images on pull requests and pushes to `dev`, but those events
cannot publish. A push requires all of the following:

1. First land this workflow on the repository's default branch (`master` at the
   time of writing). GitHub does not deliver `workflow_dispatch` events for a
   workflow that exists only on a non-default branch. Then manually dispatch it
   with `dev` selected as the run ref.
2. Set `publish` to true.
3. Supply an `experimental-*` tag.
4. Type `PUBLISH EXPERIMENTAL` exactly.
5. Type `FALKORDB SSPL DISTRIBUTION APPROVED` exactly and supply the legal
   review ticket or approval reference.
6. Type `WHOLE IMAGE DISTRIBUTION APPROVED` exactly after reviewing the full
   SBOM, third-party licenses, notices, and bundled signature data, and supply
   that review's ticket or approval reference.
7. Supply the security review ticket covering the HIGH/CRITICAL report and
   embedded-secret scan.
8. Pass the protected `dockerhub-experimental` GitHub Environment approval.
9. Configure the Docker Hub repository to enforce immutable tags matching
   `experimental-*`.

Configure that Environment with required reviewers before adding Docker Hub
credentials. Before any push, both architectures are built locally, booted,
persistence-tested, checked for embedded secrets, and gated on fixable CRITICAL
vulnerabilities. Trivy also reports every HIGH/CRITICAL finding, including
unfixed findings, for the required security review. Only the protected
per-architecture publish jobs and final manifest job can read `DOCKER_USER` and
`DOCKER_PASSWORD`. Each architecture is then pushed by canonical digest, boots
that exact digest with its default command, and repeats the vulnerability and
secret scans. The human-facing two-architecture tag is assembled only from both
passing digest artifacts, with OCI metadata plus BuildKit SBOM and provenance
attestations. The workflow never creates `latest`, checks tag absence twice,
and fails closed on registry lookup errors. Docker Hub's immutable-tag rule is
still required to eliminate the check-then-push race with an external writer.
The digest-addressed images exist in Docker Hub before the post-push exact-byte
checks finish, so use a private staging registry if even untagged candidates
must not reach the public registry.

### Publication license blocker

The repository's `autogpt_platform/LICENSE.md` uses the PolyForm Shield License
and is included in the Docker build context. Before public distribution, verify
that the final image actually contains that license and all required notices.

The image also redistributes third-party servers, system packages, language
packages, browser components, and data files. Complete a dependency-by-
dependency license and notice review from the final SBOM. In particular,
FalkorDB's server distribution uses the Server Side Public License (SSPL); its
redistribution terms require explicit legal review. Disabling its process at
runtime does not remove it from the image. An SBOM is an inventory, not a
substitute for required license texts or notices. No public Docker Hub push
should occur until the project owner signs off on this review. This is not legal
advice.

## Known limitations

- One container is one failure, maintenance, scaling, and security boundary.
  This design is not highly available and has no rolling upgrades.
- PostgreSQL, Redis, RabbitMQ, ClamAV, FalkorDB, and the application compete for
  the same CPU, memory, disk, and process limits.
- All durable services share one volume and one backup schedule. Independent
  database recovery is not yet documented or validated.
- Required email verification is incompatible with the current first-signup
  provisioning path.
- Remote deployment requires an operator-supplied TLS reverse proxy; automatic
  certificates are not bundled.
- The local `bash_exec` fallback relies on Bubblewrap user/network namespaces.
  Some Docker hosts disable the required kernel operations; use E2B or another
  supported sandbox when the release smoke command fails. Do not work around a
  failure by making the entire appliance privileged.
- Published-image size and real workload resource envelopes remain to be
  measured before the first tag.
- The workflow creates SBOM and provenance attestations but does not yet sign
  the image with a separately managed signing identity.
- The third-party license/notice set, especially FalkorDB redistribution, must
  be approved before public publication.
