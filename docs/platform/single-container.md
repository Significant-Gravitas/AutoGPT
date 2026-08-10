# Run AutoGPT in one Docker container (Experimental)

!!! warning
    This distribution is experimental and may receive breaking operational
    changes. It is intended for local evaluation and single-host self-hosting,
    with no high-availability or zero-downtime guarantee.

The image packages the AutoGPT Platform frontend and backend together with
PostgreSQL, a three-node Valkey cluster, RabbitMQ, nginx, and FalkorDB.
Only nginx listens on the container's public interface, on port `3000`.
FalkorDB-backed memory is a core part of this image and is always enabled.

All durable state is stored under `/data`. Use a named Docker volume for every
installation you intend to keep.

## Get an image

Build the image from the repository root with Docker Buildx Bake:

```bash
docker buildx bake \
  --file autogpt_platform/single-container/docker-bake.hcl \
  --load \
  single-container
```

The local build loads the image as `autogpt-platform:single-container-dev`.
Commands below use that tag. If the release you are installing documents a
supported registry reference, you may substitute its exact immutable version
tag or digest; do not guess a `latest` tag.

The image has a complete default command, so this is a valid foreground boot
check:

```bash
docker run --rm autogpt-platform:single-container-dev
```

That command does not publish the web port and uses an anonymous `/data`
volume. Use the full setup below for a usable installation.

## Quick start

Create a private environment file:

```bash
install -m 600 autogpt_platform/single-container/.env.example \
  autogpt_platform/single-container/.env
```

Edit the file and set at least:

```dotenv
AUTOGPT_PUBLIC_URL=http://localhost:3000
AUTH_ALLOW_NEW_ACCOUNTS=true
AUTH_SIGNUP_ALLOWLIST=owner@example.com
```

Replace `owner@example.com` with the intended first account. Signup is closed
by default; the temporary allowlist avoids opening registration to every email
address during bootstrap. These loopback HTTP values are only for local
evaluation. Use an HTTPS origin for LAN or remote access.

Before starting, choose a provider profile from
[Models and memory](#models-and-memory) and add it to the same file. The UI can
boot without a provider, but AutoPilot and memory model calls cannot work.

Start the appliance:

```bash
docker run --detach --name autogpt \
  --restart unless-stopped \
  --stop-timeout 360 \
  --shm-size 2g \
  --ulimit nofile=65536:65536 \
  --log-driver json-file \
  --log-opt max-size=50m \
  --log-opt max-file=5 \
  --env-file autogpt_platform/single-container/.env \
  --publish 127.0.0.1:3000:3000 \
  --volume autogpt-platform-data:/data \
  autogpt-platform:single-container-dev
```

Wait for the complete appliance to become healthy:

```bash
docker inspect --format '{{.State.Health.Status}}' autogpt
docker logs --follow autogpt
```

Open `http://localhost:3000`, create the intended account, and promote it:

```bash
docker exec autogpt autogpt-admin promote owner@example.com
```

Sign out and back in after promotion so the new session has the administrator
role.

Then set this in the environment file:

```dotenv
AUTH_ALLOW_NEW_ACCOUNTS=false
```

Apply the change by replacing only the container. Keep the same named volume:

```bash
docker stop --time 360 autogpt
docker rm autogpt
```

Repeat the `docker run` command above. Removing the container does not remove
the `autogpt-platform-data` volume.

## Port and public URL

Container port `3000` does not change. To use host port `3300`, change the run
command to:

```bash
--publish 127.0.0.1:3300:3000
```

and set:

```dotenv
AUTOGPT_PUBLIC_URL=http://localhost:3300
```

`AUTOGPT_PUBLIC_URL` must be the exact origin used in the browser, including
the scheme and any non-default port. Docker cannot discover the host-side port
mapping from inside the container. A mismatch breaks authentication actions,
callbacks, cookies, and generated links.

For a remote host, terminate TLS in a reverse proxy on the Docker host and keep
the container bound to loopback:

```dotenv
AUTOGPT_PUBLIC_URL=https://agents.example.com
```

The reverse proxy must route all paths to `127.0.0.1:3000`, support WebSocket
upgrades for `/_agpt/ws`, and allow long-lived streaming requests below
`/_agpt/api`. Do not publish PostgreSQL, Valkey, RabbitMQ, FalkorDB, or
backend service ports.

For LAN access, bind the TLS reverse proxy to the LAN interface and keep the
container published only on `127.0.0.1:3000`. Set `AUTOGPT_PUBLIC_URL` to the
matching HTTPS origin, for example `https://agents.lan.example`; do not expose
the container's plaintext port directly or leave the URL at the localhost
default.

## Account policy

New-account creation is closed by default, including on localhost. Existing
accounts can still sign in when signup is closed.

To allow only selected accounts during provisioning, set both variables:

```dotenv
AUTH_ALLOW_NEW_ACCOUNTS=true
AUTH_SIGNUP_ALLOWLIST=owner@example.com
```

The allowlist accepts exact email addresses and entries beginning with `@` for
an entire domain. It applies to email/password signup and first-time social
login because both create an account. Prefer exact addresses; use a domain
entry only for a domain you fully control, then narrow the list after
bootstrap. Setting `AUTH_ALLOW_NEW_ACCOUNTS=false` blocks all new accounts
regardless of the allowlist.

Fresh installations should keep `AUTOGPT_ENABLE_LEGACY_AUTH=false`. Enable it
only when intentionally migrating an existing legacy symmetric-JWT setup.

Required email verification is not supported by this image and intentionally
stops startup if enabled. Keep:

```dotenv
AUTH_REQUIRE_EMAIL_VERIFICATION=false
```

Postmark can provide password-reset and email-change messages:

```dotenv
POSTMARK_SERVER_API_TOKEN=
POSTMARK_SENDER_EMAIL=
POSTMARK_WEBHOOK_TOKEN=
```

Set `POSTMARK_SENDER_EMAIL` to a sender verified by your Postmark account. This
does not add account-verification support.

Social login uses the `AUTH_*` credentials in `.env.example`. Agent block OAuth
integrations use the separate unprefixed credentials. The prebuilt frontend
does not support configuring Google Picker public keys at runtime.

## Models and memory

FalkorDB and Graphiti memory are always enabled and persisted under `/data`.
FalkorDB cannot be disabled in this distribution.

The image does not include model-provider credentials. Configure one of the
following profiles before expecting AutoPilot and memory extraction to work.

### OpenRouter with OpenAI embeddings

The default remote profile uses OpenRouter for chat and memory extraction and
OpenAI for embeddings:

```dotenv
CHAT_USE_LOCAL=false
CHAT_USE_OPENROUTER=true
OPEN_ROUTER_API_KEY=YOUR_OPENROUTER_KEY
OPENAI_API_KEY=YOUR_OPENAI_KEY
```

Both keys are needed for the complete memory path. `OPENAI_API_KEY` alone does
not select direct OpenAI routing for AutoPilot.

### Anthropic chat with remote memory

To route AutoPilot directly to Anthropic:

```dotenv
CHAT_USE_LOCAL=false
CHAT_USE_OPENROUTER=false
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY
OPEN_ROUTER_API_KEY=YOUR_OPENROUTER_KEY
OPENAI_API_KEY=YOUR_OPENAI_KEY
```

The Anthropic key changes the AutoPilot chat transport. The OpenRouter and
OpenAI keys are still required by Graphiti's default remote extraction and
embedding clients.

### Ollama or another local OpenAI-compatible server

For the default local profile, install both the chat model and memory embedding
model on the Docker host:

```bash
ollama pull hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M
ollama pull nomic-embed-text
```

The chat model and exact `Q4_K_M` artifact are published in the
[Unsloth Qwen3.5-4B-GGUF repository](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main).
Keep the model identifier in the pull command and environment setting
identical.

Then set:

```dotenv
CHAT_USE_LOCAL=true
CHAT_BASE_URL=http://host.docker.internal:11434/v1
CHAT_API_KEY=ollama
CHAT_FAST_STANDARD_MODEL=hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M
```

`CHAT_API_KEY` must be non-empty even if the local server ignores it. The
local transport automatically makes Graphiti inherit the same base URL and API
key, rewrites its default extraction and reranker model to the configured local
chat model, and uses `nomic-embed-text` for embeddings. Separate
`GRAPHITI_*` routing variables are unnecessary unless you want an override.

On Docker Engine, add `--add-host host.docker.internal:host-gateway` to the run
command for this local-model profile. Docker Desktop provides that hostname
without the extra option. Small quantized models reduce memory requirements,
but latency and answer quality remain hardware-, model-, and
workload-dependent; select another compatible model when the default does not
meet your needs.

Check connectivity from the running appliance:

```bash
docker exec autogpt \
  curl --fail --show-error http://host.docker.internal:11434/api/tags
```

The same settings can point at a remote vLLM, LocalAI, LM Studio, LiteLLM, or
other OpenAI-compatible HTTPS endpoint, provided it serves both the configured
chat model and `nomic-embed-text`. Do not expose an unauthenticated model server
to the internet. See the [local LLM guide](copilot-local-llm.md) for model and
context-window guidance.

Additional provider keys consumed by backend blocks may be placed in the same
environment file. They are passed to backend roles, not indiscriminately to the
public frontend process.

## Security boundary

The browser-facing nginx and Next.js processes run under Unix identities that
are separate from backend services. The frontend receives an explicit runtime
environment allowlist and connects to PostgreSQL through a passwordless local
peer role restricted to the Better Auth tables and columns it needs. It does
not receive the PostgreSQL superuser password, RabbitMQ or Valkey passwords,
the FalkorDB password, or encryption keys.

Generated database, queue, cache, memory, encryption, authentication, signing,
secrets are created on first boot and stored in `/data/config/runtime.env` as
`root:root` mode `0600`. Reusing the named volume reuses those secrets.
Supplying a different value for a persisted secret on a later boot fails
instead of silently rotating it.

These controls limit compromise between co-located processes, but Docker daemon
administrators and anyone who can read the data volume remain fully trusted.
Treat the host environment file, `/data`, backups, and unredacted diagnostic
output as secret-bearing material.

Only port `3000` should be published. Internal AppService RPC is bound to the
container's loopback interface, and Valkey traffic requires authentication.

## Optional processes

FalkorDB is mandatory. The one supported process toggle is:

```dotenv
AUTOGPT_ENABLE_BOT_SERVICES=false
```

Bot services should remain off unless their required platform credentials and
public routes are configured. This setting stops processes; it does not make the
image smaller.

## Persistence

The named volume mounted at `/data` contains all durable appliance state:

| Path | Contents |
| --- | --- |
| `/data/config` | Generated runtime secrets and backend configuration |
| `/data/postgres` | Authentication and platform data |
| `/data/rabbitmq` | Queue state |
| `/data/valkey` | Three-node Valkey state |
| `/data/falkordb` | Graphiti memory data |
| `/data/workspaces` | User workspaces |
| `/data/home` and `/data/frontend-home` | Application home directories |
| `/data/cache` | Backend and Next.js caches |

Do not mount one volume into two running AutoGPT containers. Use a different
named volume for every installation.

To confirm which volume a container uses:

```bash
docker inspect --format \
  '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}' \
  autogpt
```

## Cold backup

Stop the appliance before archiving the coupled service state. It remains
unavailable for the duration of the archive, which grows with `/data`:

```bash
docker stop --time 360 autogpt
umask 077
touch autogpt-platform-data.tgz
chmod 600 autogpt-platform-data.tgz

docker run --rm \
  --entrypoint tar \
  --volume autogpt-platform-data:/data:ro \
  --volume "${PWD}:/backup" \
  autogpt-platform:single-container-dev \
  -czf /backup/autogpt-platform-data.tgz -C /data .
```

Restart the unchanged installation when the archive completes:

```bash
docker start autogpt
```

Record the exact image reference or digest, environment file, Git commit, and
backup checksum beside the archive. The archive is plaintext and contains user
content, provider credentials, auth keys, and database passwords. Encrypt it
with an approved backup mechanism and remove unencrypted staging copies.

## Restore into a new volume

Restore into a new named volume so the source remains recoverable:

```bash
RESTORE_VOLUME="autogpt-platform-data-restored-$(date -u +%Y%m%dT%H%M%SZ)"
if docker volume inspect "${RESTORE_VOLUME}" >/dev/null 2>&1; then
  echo "Refusing to reuse existing volume: ${RESTORE_VOLUME}" >&2
  exit 1
fi
docker volume create "${RESTORE_VOLUME}"

docker run --rm \
  --entrypoint tar \
  --volume "${RESTORE_VOLUME}:/data" \
  --volume "${PWD}:/backup:ro" \
  autogpt-platform:single-container-dev \
  -xzf /backup/autogpt-platform-data.tgz -C /data
```

Validate the restored layout without starting application services or allowing
network access:

```bash
docker run --rm \
  --network none \
  --entrypoint /bin/sh \
  --volume "${RESTORE_VOLUME}:/data:ro" \
  autogpt-platform:single-container-dev \
  -ceu '
    test -s /data/config/runtime.env
    test -s /data/config/backend.json
    test -s /data/postgres/PG_VERSION
    test -d /data/rabbitmq/mnesia
    test -d /data/valkey/17000
    test -d /data/falkordb
  '
```

This structural check does not prove that each database can start. A full
recovery rehearsal boots live schedules, stored credentials, and executors,
and some services fetch runtime data during startup. Perform it only on a
dedicated egress-filtered host or network after revoking or replacing
production provider and integration credentials. There is no generic appliance
switch that safely disables every possible outbound action. Retain both
volumes until the restore is accepted.

Never selectively mix service directories from different backups.

## Upgrade and rollback

Before an upgrade:

1. Record the running image reference and image ID.
2. Stop the container and take a cold backup.
3. Pull or build the new image.
4. Remove only the stopped container.
5. Repeat the Quick start run command with the same environment file and named
   volume but the new image reference.
6. Wait for full health, then test login, memory, one agent execution, streaming,
   WebSockets, and persistence across one restart.

Useful image evidence is available with:

```bash
docker inspect --format '{{.Config.Image}} {{.Image}}' autogpt
```

Startup applies database migrations before publishing readiness. Do not run an
older image against a volume already migrated by a newer image. Rollback means
running the prior image with its matching pre-upgrade archive restored into a
new volume.

## Health and troubleshooting

Docker health checks every bundled dependency and application role:

```bash
docker inspect --format '{{.State.Health.Status}}' autogpt
docker exec autogpt autogpt-healthcheck
docker logs --tail 500 autogpt
```

`GET /healthz` checks nginx only; it is not proof that the whole appliance is
ready.

| Symptom | What to check |
| --- | --- |
| The browser cannot connect after `docker run IMAGE` | A bare run does not publish a port. Use the complete Quick start command. |
| Port `3300` opens but auth actions fail | Use `--publish 127.0.0.1:3300:3000` and set `AUTOGPT_PUBLIC_URL=http://localhost:3300`, then replace the container. |
| Signup says registration is closed | Temporarily enable signup with an exact-address `AUTH_SIGNUP_ALLOWLIST`, replace the container, create the intended accounts, and close signup again. |
| The container remains `starting` or becomes `unhealthy` | First boot can take several minutes. Run `autogpt-healthcheck` and inspect container logs for the first failed service. |
| AutoPilot returns a provider `401` | Configure the key for the selected transport. The default remote route needs `OPEN_ROUTER_API_KEY`; complete remote memory also needs `OPENAI_API_KEY`. |
| Local chat works but memory ingestion fails | Install `nomic-embed-text` on the configured local server and confirm its `/v1/embeddings` endpoint works. |
| Ollama cannot be reached | Keep the host-gateway option, ensure Ollama listens on an address Docker can reach, and test `/api/tags` from inside the container. |
| The container exits after a persistent health failure | The watchdog intentionally stops the appliance. Keep `--restart unless-stopped` so Docker can recover it. |
| Data appears missing after replacement | The new container is using another or anonymous `/data` volume. Inspect its mount and reattach the original named volume. |

## Known limitations

- One container is one failure, maintenance, scaling, and security boundary.
- PostgreSQL, Valkey, RabbitMQ, FalkorDB, browser tooling, and the
  application compete for the same host resources.
- All durable services share one volume and one backup schedule.
- Uploaded files are not scanned for malware. Unlike the hosted platform, this
  image bundles no antivirus daemon, so treat every upload as trusted input.
- Required email verification is unsupported.
- Remote TLS termination is operator-supplied.
- The local `bash_exec` fallback depends on host support for Bubblewrap user and
  network namespaces. Do not make the entire appliance privileged to work
  around a host that disables them.
- Registry availability is documented per release; otherwise build the
  validated `linux/amd64` or `linux/arm64` image from source as described above.
