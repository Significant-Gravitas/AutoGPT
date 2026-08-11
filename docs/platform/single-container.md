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

Published single-container images begin with the first stable release after the
publication workflow is enabled; earlier Platform releases are not backfilled.
After that release completes, use the most recent fully verified stable image:

```bash
IMAGE=significantgravitas/autogpt:latest
docker pull "${IMAGE}"
```

For a reproducible installation, replace `latest` with an immutable `vX.Y.Z`
tag or manifest digest. Docker image tags map to source releases as follows:

- `latest` points to the newest stable AutoGPT Platform release.
- `vX.Y.Z` is the immutable image for GitHub release
  `autogpt-platform-beta-vX.Y.Z`.
- `sha-<git-sha>` is the immutable image for an exact `dev` or release source
  revision.

To build from source instead, run Docker Buildx Bake from the repository root:

```bash
docker buildx bake \
  --file autogpt_platform/single-container/docker-bake.hcl \
  --load \
  single-container
IMAGE=autogpt-platform:single-container-dev
```

Commands below use the shell variable `IMAGE`, whether it identifies the
Docker Hub image, a pinned digest, or the local build.

The image has a complete default command, so this is a valid foreground boot
check:

```bash
docker run --rm "${IMAGE}"
```

That command does not publish the web port and uses an anonymous `/data`
volume. Use the full setup below for a usable installation.

## Quick start

Create a private environment file:

```bash
umask 077
touch autogpt.env
chmod 600 autogpt.env
```

When working from a source checkout, you can copy
`autogpt_platform/single-container/.env.example` instead to see every optional
setting.

Edit the file and set at least:

```dotenv
AUTOGPT_PUBLIC_URL=http://localhost:3000
```

Signup starts open so a fresh installation can create its first account. The
run command below binds the app only to loopback. If other users can reach the
URL, anyone can register until you close signup. To limit provisioning to one
address, optionally set `AUTH_SIGNUP_ALLOWLIST=owner@example.com` before the
first boot. Configure an HTTPS origin before creating real accounts or entering
credentials on any LAN or remote deployment.

Provider keys are not required to boot, create an account, use Builder, or run
provider-free blocks. Model-backed functions return their normal actionable
missing-credential error until you configure a profile from
[Models and memory](#models-and-memory).

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
  --env-file autogpt.env \
  --publish 127.0.0.1:3000:3000 \
  --volume autogpt-platform-data:/data \
  "${IMAGE}"
```

Wait for the complete appliance to become healthy:

```bash
docker inspect --format '{{.State.Health.Status}}' autogpt
docker logs --follow autogpt
```

Test installations used about 5–6 GiB of memory during startup and steady-state
health checks. This is measured guidance, not a guaranteed minimum; allow
headroom for enabled services, agents, local models, and workload growth.

Open `http://localhost:3000`, create the intended account, and promote it:

```bash
docker exec autogpt autogpt-admin promote owner@example.com
```

Replace `owner@example.com` with the email address of the account you created.
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

New-account creation starts open so the first administrator can sign up.
Existing accounts can still sign in after signup is closed.

To allow only selected accounts during provisioning, keep signup enabled and
set an allowlist:

```dotenv
AUTH_ALLOW_NEW_ACCOUNTS=true
AUTH_SIGNUP_ALLOWLIST=owner@example.com
```

The allowlist accepts exact email addresses and entries beginning with `@` for
an entire domain. It applies to email/password signup and first-time social
login because both create an account. Prefer exact addresses; use a domain
entry only for a domain you fully control, then narrow the list after
bootstrap. Domain matching trusts the identity provider's asserted email;
public email domains such as `@gmail.com` are not safe allowlist entries.
Setting `AUTH_ALLOW_NEW_ACCOUNTS=false` blocks all new accounts regardless of
the allowlist; recreate the container with the same volume to apply the setting
after promoting the intended administrator.

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

The image does not include model-provider credentials. This does not block
startup, authentication, Builder, or provider-free blocks. Configure one of
the following profiles before expecting AutoPilot and memory extraction to
work; until then, provider-backed requests return the same missing-credential
errors as other AutoGPT deployment modes.

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

Generated database, queue, cache, memory, encryption, authentication, and
signing secrets are created on first boot and stored in
`/data/config/runtime.env` as `root:root` mode `0600`. Reusing the named volume
reuses those secrets.
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

Stop the appliance before archiving the coupled service state. The block below
uses the stopped container's exact local image ID, writes to a unique partial
file, and promotes it to the final timestamped name only after `tar` succeeds.
The appliance remains unavailable for the duration of the archive, which grows
with `/data`:

```bash
(
  set -euo pipefail
  BACKUP_IMAGE="$(docker inspect --format '{{.Image}}' autogpt)"
  BACKUP_VOLUME="$(docker inspect --format \
    '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}' \
    autogpt)"
  BACKUP_DIR="${PWD}/autogpt-backups"
  BACKUP_FILE="autogpt-platform-data-$(date -u +%Y%m%dT%H%M%SZ).tgz"
  PARTIAL_FILE="${BACKUP_FILE}.partial"
  # Invoked by the EXIT trap below.
  # shellcheck disable=SC2329
  restart_autogpt() {
    local exit_status="$1"
    trap - EXIT
    if ! docker start autogpt >/dev/null; then
      echo "Backup finished but the autogpt container could not restart" >&2
      exit_status=1
    fi
    exit "${exit_status}"
  }

  : "${BACKUP_VOLUME:?Container has no named volume mounted at /data}"
  if [[ "$(docker inspect --format '{{.State.Running}}' autogpt)" != true ]]; then
    echo "Refusing backup because the autogpt container is not running" >&2
    exit 1
  fi
  mkdir -p "${BACKUP_DIR}"
  chmod 700 "${BACKUP_DIR}"
  if [[ -e "${BACKUP_DIR}/${BACKUP_FILE}" || \
        -e "${BACKUP_DIR}/${PARTIAL_FILE}" ]]; then
    echo "Refusing to overwrite an existing backup: ${BACKUP_FILE}" >&2
    exit 1
  fi

  trap 'restart_autogpt "$?"' EXIT
  docker stop --time 360 autogpt
  umask 077
  touch "${BACKUP_DIR}/${PARTIAL_FILE}"
  chmod 600 "${BACKUP_DIR}/${PARTIAL_FILE}"

  docker run --rm \
    --entrypoint tar \
    --volume "${BACKUP_VOLUME}:/data:ro" \
    --volume "${BACKUP_DIR}:/backup" \
    "${BACKUP_IMAGE}" \
    -czf "/backup/${PARTIAL_FILE}" -C /data .

  mv "${BACKUP_DIR}/${PARTIAL_FILE}" "${BACKUP_DIR}/${BACKUP_FILE}"
  printf 'Backup written to %s with image %s\n' \
    "${BACKUP_DIR}/${BACKUP_FILE}" "${BACKUP_IMAGE}"
)
```

The exit trap restarts the unchanged installation after the archive succeeds or
if a later backup command fails. Verify that it is running again:

```bash
docker inspect --format '{{.State.Status}}' autogpt
```

Record the exact image reference or digest, environment file, Git commit, and
backup checksum beside the archive. The archive is plaintext and contains user
content, provider credentials, auth keys, and database passwords. Encrypt it
with an approved backup mechanism and remove unencrypted staging copies.

## Restore into a new volume

Restore into a new named volume so the source remains recoverable. Set
`BACKUP_FILE` to the timestamped archive name and `RESTORE_IMAGE` to the
immutable tag or digest recorded with that backup. If the archive is not under
`./autogpt-backups`, also set `BACKUP_DIR` before running the block:

```bash
(
  set -euo pipefail
  : "${BACKUP_FILE:?Set BACKUP_FILE to the timestamped archive filename}"
  : "${RESTORE_IMAGE:?Set RESTORE_IMAGE to the recorded immutable image}"
  BACKUP_DIR="${BACKUP_DIR:-${PWD}/autogpt-backups}"
  RESTORE_VOLUME="autogpt-platform-data-restored-$(date -u +%Y%m%dT%H%M%SZ)"

  if [[ "${BACKUP_FILE}" == */* ]]; then
    echo "BACKUP_FILE must be a filename within BACKUP_DIR" >&2
    exit 1
  fi
  if [[ ! -f "${BACKUP_DIR}/${BACKUP_FILE}" ]]; then
    echo "Backup archive does not exist: ${BACKUP_DIR}/${BACKUP_FILE}" >&2
    exit 1
  fi
  if docker volume inspect "${RESTORE_VOLUME}" >/dev/null 2>&1; then
    echo "Refusing to reuse existing volume: ${RESTORE_VOLUME}" >&2
    exit 1
  fi
  docker volume create "${RESTORE_VOLUME}"

  docker run --rm \
    --entrypoint tar \
    --volume "${RESTORE_VOLUME}:/data" \
    --volume "${BACKUP_DIR}:/backup:ro" \
    "${RESTORE_IMAGE}" \
    -xzf "/backup/${BACKUP_FILE}" -C /data

  printf 'Restored %s into volume %s\n' \
    "${BACKUP_FILE}" "${RESTORE_VOLUME}"
)
```

Validate the restored layout without starting application services or allowing
network access. Set `RESTORE_VOLUME` to the volume printed above and reuse the
same `RESTORE_IMAGE`:

```bash
(
  set -euo pipefail
  : "${RESTORE_VOLUME:?Set RESTORE_VOLUME to the restored volume name}"
  : "${RESTORE_IMAGE:?Set RESTORE_IMAGE to the recorded immutable image}"

  docker run --rm \
    --network none \
    --entrypoint /bin/sh \
    --volume "${RESTORE_VOLUME}:/data:ro" \
    "${RESTORE_IMAGE}" \
    -ceu '
      test -s /data/config/runtime.env
      test -s /data/config/backend.json
      test -s /data/postgres/PG_VERSION
      test -d /data/rabbitmq/mnesia
      test -d /data/valkey/17000
      test -d /data/valkey/17001
      test -d /data/valkey/17002
      test -d /data/falkordb
    '
)
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
| Signup says registration is closed | Set `AUTH_ALLOW_NEW_ACCOUNTS=true` with an exact-address `AUTH_SIGNUP_ALLOWLIST=owner@example.com`, replace the container, create the intended accounts, and close signup again. |
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
