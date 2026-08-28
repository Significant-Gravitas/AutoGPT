# Run AutoGPT in One Docker Container (Experimental)

The image packages the AutoGPT Platform frontend and backend together with
PostgreSQL, a three-node Valkey cluster, RabbitMQ, nginx, and FalkorDB.
Only nginx listens on the container's public interface, on port `3000`.
The FalkorDB service is a core part of this image and always runs; Graphiti
memory is enabled by the image's default configuration and still requires a
working model-provider profile.

All durable state is stored under `/data`. Use a named Docker volume for every
installation you intend to keep.

## Requirements

- Docker Engine or Docker Desktop capable of running Linux containers on
  `linux/amd64` or `linux/arm64`. Building locally also requires Docker Buildx.
- More than the measured 5–6 GiB of appliance memory use, with additional
  headroom for agents, local models, and workload growth.
- Host storage with enough capacity and throughput for the image, the growing
  `/data` volume, and at least one complete compressed backup archive alongside
  the live data. Shutdown and backup time depend on storage throughput and fsync
  latency; SSD-class storage is recommended but not a measured support floor.

There is no measured minimum CPU count or supported concurrent-run ceiling;
performance depends on the enabled services and workload.

## Get an image

Use `latest` for the most recent fully verified stable AutoGPT Platform image:

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

Older `canary-sha-*` tags are legacy pre-release validation artifacts, not a
currently published or supported tag family.

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

The image has a complete default command, so you can optionally perform a
foreground boot check:

```bash
docker run --rm "${IMAGE}"
```

Skip this check if you want to proceed directly to a persistent installation;
Quick start performs the same first-boot work against its named volume. The
check runs in the foreground. First boot may take several minutes and use
roughly 5–6 GiB of memory; press Ctrl-C to stop it. That interruption is safe
for this check because `--rm` discards its anonymous `/data` volume. Do not
generalize it to an installation that uses a named volume. The command does not
publish the web port. Use the full setup below for a usable installation.

## Quick start

Create a private environment file:

```bash
umask 077
touch autogpt.env
chmod 600 autogpt.env
```

When working from a source checkout, you can copy
`autogpt_platform/single-container/.env.example` instead to see common optional
settings. The backend-only `BEHAVE_AS` control and its limits are documented in
[Security boundary](#security-boundary).

Edit the file and set at least the public URL and exact address for the first
account:

```dotenv
AUTOGPT_PUBLIC_URL=http://localhost:3000
AUTH_SIGNUP_ALLOWLIST=owner@example.com
```

Replace `owner@example.com` with the email address for the intended first
account. Passwords must contain at least 12 characters. Exact-address
allowlisting compares the email string asserted during signup; because this
image does not support required email verification, it is not proof of mailbox
ownership. New-account creation defaults open when the allowlist is empty; the
value above immediately restricts signup to that address. If the allowlist is
omitted, anyone who can reach the published port can create an account until
`AUTH_ALLOW_NEW_ACCOUNTS=false` is applied. The run command below binds the app
only to loopback. Create and promote the administrator and close signup while
the app is still loopback-only. Configure an HTTPS origin before entering
credentials on any LAN or remote deployment.

Provider keys are not required to boot, create an account, use Builder, or run
provider-free blocks. Model-backed functions return their normal actionable
missing-credential error until you configure a profile from
[Models and memory](#models-and-memory).

Start the appliance:

```bash
docker run --detach --name autogpt \
  --restart unless-stopped \
  --shm-size 2g \
  --ulimit nofile=65536:65536 \
  --log-driver json-file \
  --log-opt max-size=50m \
  --log-opt max-file=5 \
  --env-file autogpt.env \
  --publish 127.0.0.1:3000:3000 \
  --volume autogpt-data:/data \
  "${IMAGE}"
```

The `--shm-size 2g` allocation keeps temporary ChatGPT/Codex authentication
homes in memory instead of the container's writable layer. The `nofile` limit
sets a predictable per-process file-descriptor ceiling for the bundled services.
The JSON log options retain about five 50 MB files instead of allowing container
logs to grow without a bound.

Wait for the complete appliance to become healthy:

```bash
docker inspect --format '{{.State.Health.Status}}' autogpt
docker logs --tail 100 --follow autogpt
```

Do not stop the container while the first boot is applying database migrations.
If a migration is interrupted, later boots refuse to continue and the logs
identify the migration and recovery choices. For a brand-new empty installation,
remove that installation's unused `/data` volume and start again. For an existing
installation, restore the pre-upgrade backup. Use `prisma migrate resolve` only
after determining whether that migration's changes reached the database.

For an empty first boot that used the exact Quick start names, first verify that
the volume contains no user data, then discard that failed installation:

```bash
docker stop autogpt
docker rm autogpt
docker volume rm autogpt-data
```

Repeat Quick start afterward. The failed appliance can remain in a restart loop
until explicitly stopped, so `docker exec` is not a reliable way to run
`prisma migrate resolve`; that advanced path requires PostgreSQL running against
the affected volume from a separate recovery environment. Do not use it without
inspecting whether the migration's database changes completed.

Test installations used about 5–6 GiB of memory during startup and steady-state
health checks. This is measured guidance, not a guaranteed minimum; allow
headroom for enabled services, agents, local models, and workload growth. On
Docker Desktop, make sure the VM's memory allocation in **Settings → Resources**
exceeds that observed use and leaves the same headroom.

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
docker stop autogpt
docker rm autogpt
```

Repeat the `docker run` command above. Removing the container does not remove
the `autogpt-data` volume.

## Stopping

`docker stop autogpt` is designed and tested to complete inside Docker's stock
10-second timeout, so no host-wide timeout change is needed; a longer host
timeout does not extend the internal Supervisor caps. The shipped one-second
runtime, five-second state, and one-second event-listener phases measured about
8.4 seconds in the shutdown test setup. That is a narrow measured margin, not a
graceful-shutdown guarantee for slower storage, so do not shorten Docker's
timeout. Runtime processes are signaled first. PostgreSQL, RabbitMQ, Valkey,
and FalkorDB are signaled afterward and each gets at most five seconds to exit
before Supervisor forces it down. The event listener stops last. Larger or
slower state may require normal crash recovery on the next boot.

Agent runs still executing when the container stops are abandoned. Their queue
messages can be dropped and their execution rows can remain `RUNNING`, but they
do not resume. Start a new run after restart.

Supervisor process names are group-qualified. Use `supervisorctl status` to see
names such as `runtime:rest` and `state:postgres`.

## Port and public URL

Container port `3000` does not change. To use host port `3300`, change the run
command to:

```text
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

For LAN or remote access, keep AutoGPT bound to loopback and place it behind a
TLS reverse proxy running outside the AutoGPT container. The proxy provides
HTTPS and forwards requests to port `3000`; AutoGPT handles application routing
internally.

Before exposing the proxy, complete the [Account policy](#account-policy)
bootstrap, promote the intended administrator, and close new-account signup.
The bundled nginx records the external proxy as the immediate client, so the
proxy must preserve its own client-attribution logs and enforce any per-client
IP rate limits or access rules.

Set `AUTOGPT_PUBLIC_URL` to the browser-visible HTTPS origin, for example:

```dotenv
AUTOGPT_PUBLIC_URL=https://agents.example.com
```

Do not expose the container's plaintext port directly or leave the URL at the
localhost default.

## Account policy

New-account creation defaults open so the first administrator can sign up. A
nonempty allowlist immediately restricts who can create an account. Existing
accounts can still sign in after signup is closed.

To allow only selected accounts during provisioning, keep signup enabled and
set an allowlist:

```dotenv
AUTH_ALLOW_NEW_ACCOUNTS=true
AUTH_SIGNUP_ALLOWLIST=owner@example.com
```

The allowlist accepts exact email addresses and entries beginning with `@` for
an entire domain. Separate multiple entries with commas, for example
`AUTH_SIGNUP_ALLOWLIST=owner@example.com,teammate@example.com`. It applies to
email/password signup and any first-time account creation through a configured
provider endpoint; the bundled UI does not show social-login buttons. Prefer
exact addresses; use a domain entry only for a domain you fully control, then
narrow the list after bootstrap. Domain matching trusts the identity provider's
asserted email; public email domains such as `@gmail.com` are not safe allowlist
entries.

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
does not add account-verification support. Without these Postmark settings,
there is no self-service password-reset email or appliance CLI for resetting a
password. Store the administrator password securely before closing signup.

### Add or recover an administrator

To add an account after signup is closed, temporarily bind the appliance to
loopback, set `AUTH_ALLOW_NEW_ACCOUNTS=true`, and set
`AUTH_SIGNUP_ALLOWLIST` to that account's exact email address. Replace the
container with the same named volume and original launch options, create the
account, and promote it:

```bash
docker exec autogpt autogpt-admin promote new-owner@example.com
```

Then set `AUTH_ALLOW_NEW_ACCOUNTS=false` and replace the container again. If an
administrator password is lost and Postmark password reset was not configured,
this procedure creates a replacement administrator; it does not reset the
existing account's password. Keep the port loopback-only throughout recovery,
or enforce equivalent HTTPS and network access controls before reopening
signup.

Setting both `AUTH_*_CLIENT_ID` and `AUTH_*_CLIENT_SECRET` values for a social
provider registers reachable OAuth sign-in and callback endpoints. The bundled
local-mode frontend does not render buttons for them, so leave each pair empty
unless you intentionally use that direct provider flow. `AUTH_SIGNUP_ALLOWLIST`
and `AUTH_ALLOW_NEW_ACCOUNTS` still gate first-time account creation through
those endpoints. Agent block OAuth integrations use the separate unprefixed
credentials. The prebuilt frontend does not support configuring Google Picker
public keys at runtime.

## Models and memory

The FalkorDB service always runs and persists under `/data`; it has no supported
process toggle in this distribution. Graphiti memory is enabled by the image's
default feature configuration.

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

Configure the host's Ollama service with `OLLAMA_CONTEXT_LENGTH=32768` and
restart Ollama before using AutoPilot. This is an Ollama-server setting, not an
`autogpt.env` entry; Ollama's smaller default context cannot hold AutoPilot's
roughly 8k-token system prompt.

The chat model and exact `Q4_K_M` artifact are published in the
[Unsloth Qwen3.5-4B-GGUF repository](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main).
Keep the model identifier in the pull command and environment setting
identical. The 4B model is the smaller default chosen for this all-in-one
appliance's shared memory budget; larger models remain optional when the host
has sufficient resources.

Then set:

```dotenv
CHAT_USE_LOCAL=true
CHAT_BASE_URL=http://host.docker.internal:11434/v1
CHAT_API_KEY=ollama
CHAT_FAST_STANDARD_MODEL=hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M
```

`CHAT_API_KEY` must be non-empty even if the local server ignores it. The
local transport makes Graphiti inherit the same base URL and API key. With the
default profile above, Graphiti rewrites its extraction and reranker models to
`hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M` and uses `nomic-embed-text` for
embeddings. If you choose another chat model, also set `GRAPHITI_LLM_MODEL`
and `GRAPHITI_RERANKER_MODEL` to a model that the endpoint serves. Set
`GRAPHITI_EMBEDDER_MODEL` too when its embedding model uses another slug.

On Docker Engine, add `--add-host host.docker.internal:host-gateway` to the run
command for this local-model profile. Docker Desktop provides that hostname
without the extra option. To apply the flag to an existing container, stop and
remove the container, then repeat the Quick start command with the same named
volume and the added flag. Small quantized models reduce memory requirements,
but latency and answer quality remain hardware-, model-, and workload-dependent;
select another compatible model when the default does not meet your needs.

Check connectivity from the running appliance:

```bash
docker exec autogpt \
  curl --fail --show-error http://host.docker.internal:11434/api/tags
```

The same settings can point at a remote vLLM, LocalAI, LM Studio, LiteLLM, or
other OpenAI-compatible HTTPS endpoint, provided it serves every configured
chat and Graphiti model slug. Do not expose an unauthenticated model server to
the internet. See the
[AutoPilot local-LLM guide](copilot-local-llm.md)
for model and context-window guidance.

Additional provider keys consumed by backend blocks may be placed in the same
environment file. Most remain backend-only. The Next.js server process always
receives the required `BETTER_AUTH_SECRET` and can receive configured `AUTH_*`
provider client secrets, `OPENAI_API_KEY`, `TRANSCRIPTION_API_KEY`, and the
legacy `SUPABASE_JWT_SECRET`. These values remain server-side process
environment and are not baked into the browser bundle.

### Database connection tuning

`DB_CONNECTION_LIMIT` controls each backend role's Prisma connection pool and
accepts `1` through `5` (default `5`). `DB_CONNECT_TIMEOUT` controls connection
setup and accepts `1` through `600` seconds (default `60`), while
`DB_POOL_TIMEOUT` controls how long a request can wait for a pooled connection
and accepts `1` through `3600` seconds (default `300`). All roles share the
bundled PostgreSQL instance, so do not raise per-role limits beyond the enforced
range. If requests stall under concurrent runs, inspect service and PostgreSQL
logs for pool exhaustion before changing these values.

## Security boundary

`BEHAVE_AS` defaults to `local`, which bypasses subscription entitlement gating
for policies that opt into a local exemption, which currently includes every
defined entitlement. `local` is a broader product behavior profile: it can also
enable blocks intentionally disabled in hosted mode, select self-host model
routing, and change diagnostics or telemetry behavior. This is appropriate for
single-tenant self-hosting. Any
multi-tenant or hosted deployment must set `BEHAVE_AS=cloud` for backend
entitlement enforcement, but that setting is not a complete hosted-mode switch:
the bundled frontend is compiled in local mode. Cloud mode also expects the
hosted model catalog, subscription tiers, and payment controls. Do not treat
this image as a turnkey multi-tenant hosted distribution. Use the supported
cloud deployment and build, and review its full security boundary.

The browser-facing nginx and Next.js processes run under Unix identities that
are separate from backend services. The frontend receives an explicit runtime
environment allowlist and connects to PostgreSQL through a passwordless local
peer role restricted to the Better Auth tables and columns it needs. It does
receive `BETTER_AUTH_SECRET`, configured `AUTH_*` social-login client secrets,
and optional OpenAI, transcription, and legacy-JWT values because the Next.js
server uses them. Those values remain server-side and are not baked into the
browser bundle. The frontend does not receive the PostgreSQL superuser password,
RabbitMQ or Valkey passwords, the FalkorDB password, or encryption keys.

Generated database, queue, cache, memory, encryption, authentication, and
signing secrets are created on first boot and stored in
`/data/config/runtime.env` as `root:root` mode `0600`. Reusing the named volume
reuses those secrets. Supplying a different value for a persisted secret on a
later boot fails instead of silently rotating it.

### Suspected secret exposure

There is no supported in-place rotation for the generated secrets in
`/data/config/runtime.env`. If that file or a plaintext backup is exposed,
isolate the installation and revoke external provider and OAuth credentials.
Create a replacement installation on a new volume, transfer only non-secret
agent definitions through supported export/import flows, and reconnect
credentials. Every backup from the old installation contains the same generated
secrets, so restoring an older one does not rotate them. Do not edit
`runtime.env`, selectively mix state directories, or treat a same-installation
backup as secret-exposure recovery.

These controls limit compromise between co-located processes, but Docker daemon
administrators and anyone who can read the data volume remain fully trusted.
Treat the host environment file, `/data`, backups, and unredacted diagnostic
output as secret-bearing material.

Only port `3000` should be published. Internal AppService RPC is bound to the
container's loopback interface, and Valkey traffic requires authentication.
On Linux, Docker-managed forwarding can bypass firewall policy expressed only
through tools such as UFW. The loopback address in
`--publish 127.0.0.1:3000:3000` is the exposure control; changing it to a
non-loopback address can expose the app regardless of an INPUT-chain rule.

## Optional processes

FalkorDB is mandatory. Bot services and their internal linking manager default
off. After configuring a supported chat adapter and its public routes, enable
both optional processes with:

```dotenv
AUTOGPT_ENABLE_BOT_SERVICES=true
```

For Discord, configure `AUTOPILOT_BOT_DISCORD_TOKEN`. Telegram requires both
`AUTOPILOT_BOT_TELEGRAM_TOKEN` and
`AUTOPILOT_BOT_TELEGRAM_WEBHOOK_SECRET`, plus a public HTTPS webhook registered
with Telegram. Leave the toggle `false` until the chosen adapter credentials
and routes are ready. This setting stops processes; it does not make the image
smaller.

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
| `/data/cache` | Regenerable backend and Next.js caches (excluded from backups) |

Do not mount one volume into two running AutoGPT containers. Use a different
named volume for every installation.

To confirm which volume a container uses:

```bash
docker inspect --format \
  '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}' \
  autogpt
```

## Cold backup

The block below stops the running appliance before archiving its coupled
service state. It uses the running container's exact local image ID captured
before the stop, writes to a unique partial file, and promotes it to the final
timestamped name only after `tar` succeeds. This produces a stopped-volume
snapshot. If a state service exceeds its five-second shutdown cap, the snapshot
reflects a crash stop rather than a fully graceful shutdown. Rehearse
restoration and verify service recovery before relying on it.
The appliance remains unavailable while `tar` creates and gzip-compresses the
archive; duration depends on `/data`, host storage, and CPU performance. Before
stopping it, ensure the host filesystem that contains `BACKUP_DIR` has room for
a complete compressed archive. The block enforces owner-only mode `0700` on
that directory, including when it already exists:

Start the appliance and verify full health before backing it up. The block
intentionally refuses an already stopped or crashed appliance because it must
own the coordinated transition from running to stopped state.

There is no supported consistent hot-backup procedure for this coupled volume.
If this outage is unacceptable, use a distributed deployment and the
service-native backup procedures for its independently managed data services.

```bash
(
  set -euo pipefail
  BACKUP_IMAGE_ID="$(docker inspect --format '{{.Image}}' autogpt)"
  BACKUP_IMAGE_REF="$(docker inspect --format '{{.Config.Image}}' autogpt)"
  BACKUP_IMAGE_DIGEST="$(docker image inspect --format \
    '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' \
    "${BACKUP_IMAGE_ID}")"
  BACKUP_VOLUME="$(docker inspect --format \
    '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}' \
    autogpt)"
  BACKUP_DIR="${BACKUP_DIR:-${PWD}/autogpt-backups}"
  RESTART_AFTER_BACKUP="${RESTART_AFTER_BACKUP:-true}"
  BACKUP_COMPLETE=false
  ARTIFACTS_STARTED=false
  LOCK_HELD=false
  MANAGE_CONTAINER=false
  BACKUP_FILE=''
  PARTIAL_FILE=''
  CHECKSUM_FILE=''
  CHECKSUM_PARTIAL=''

  if [[ "${RESTART_AFTER_BACKUP}" != true && \
        "${RESTART_AFTER_BACKUP}" != false ]]; then
    echo "RESTART_AFTER_BACKUP must be true or false" >&2
    exit 1
  fi
  : "${BACKUP_VOLUME:?Container has no named volume mounted at /data}"
  BACKUP_VOLUME_LABELS="$(docker volume inspect --format '{{json .Labels}}' \
    "${BACKUP_VOLUME}")"
  if [[ "${BACKUP_VOLUME_LABELS}" == *'"com.docker.volume.anonymous"'* || \
        "${BACKUP_VOLUME}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Refusing backup because /data uses an anonymous volume" >&2
    exit 1
  fi
  mkdir -p "${BACKUP_DIR}"
  BACKUP_DIR="$(cd "${BACKUP_DIR}" && pwd -P)"
  chmod 700 "${BACKUP_DIR}"
  LOCK_CONTAINER=autogpt-backup-lock

  # Invoked by the EXIT and signal traps below.
  # shellcheck disable=SC2329
  finish_backup() {
    local exit_status="$1"
    local container_running
    trap - EXIT HUP INT TERM
    if [[ "${exit_status}" -ne 0 && "${ARTIFACTS_STARTED}" == true && \
          "${BACKUP_COMPLETE}" != true ]]; then
      if ! rm -f \
        "${BACKUP_DIR}/${PARTIAL_FILE}" \
        "${BACKUP_DIR}/${CHECKSUM_PARTIAL}" \
        "${BACKUP_DIR}/${BACKUP_FILE}" \
        "${BACKUP_DIR}/${CHECKSUM_FILE}"; then
        echo "Backup failed and partial files could not be removed" >&2
        exit_status=1
      else
        echo "Backup failed; no incomplete backup artifacts were kept" >&2
      fi
    fi
    container_running=''
    if [[ "${MANAGE_CONTAINER}" == true ]]; then
      container_running="$(docker inspect --format '{{.State.Running}}' \
        autogpt 2>/dev/null || true)"
    fi
    if [[ "${MANAGE_CONTAINER}" == true && \
          ( "${exit_status}" -ne 0 || \
            "${RESTART_AFTER_BACKUP}" == true ) ]]; then
      if [[ "${container_running}" != true ]] && \
         ! docker start autogpt >/dev/null; then
        echo "AutoGPT is not running; automatic restart failed" >&2
        exit_status=1
      fi
    fi
    if [[ "${LOCK_HELD}" == true ]] && \
       ! docker rm --volumes "${LOCK_CONTAINER}" >/dev/null; then
      echo "Backup finished but its lock container could not be removed" >&2
      exit_status=1
    fi
    exit "${exit_status}"
  }

  trap 'finish_backup "$?"' EXIT
  trap 'finish_backup 129' HUP
  trap 'finish_backup 130' INT
  trap 'finish_backup 143' TERM

  if docker create --name "${LOCK_CONTAINER}" \
    --entrypoint /bin/true "${BACKUP_IMAGE_ID}" >/dev/null; then
    LOCK_HELD=true
  else
    echo "Refusing backup because another backup may be running or its lock is stale" >&2
    exit 1
  fi
  BACKUP_FILE="autogpt-data-$(date -u +%Y%m%dT%H%M%SZ).tgz"
  PARTIAL_FILE="${BACKUP_FILE}.partial"
  CHECKSUM_FILE="${BACKUP_FILE}.sha256"
  CHECKSUM_PARTIAL="${CHECKSUM_FILE}.partial"
  if [[ -e "${BACKUP_DIR}/${BACKUP_FILE}" || \
        -e "${BACKUP_DIR}/${PARTIAL_FILE}" || \
        -e "${BACKUP_DIR}/${CHECKSUM_FILE}" || \
        -e "${BACKUP_DIR}/${CHECKSUM_PARTIAL}" ]]; then
    echo "Refusing to overwrite an existing backup: ${BACKUP_FILE}" >&2
    exit 1
  fi
  if [[ "$(docker inspect --format '{{.State.Running}}' autogpt)" != true ]]; then
    echo "Refusing backup because the autogpt container is not running" >&2
    exit 1
  fi

  MANAGE_CONTAINER=true
  docker stop autogpt
  umask 077
  ARTIFACTS_STARTED=true
  touch "${BACKUP_DIR}/${PARTIAL_FILE}"
  chmod 600 "${BACKUP_DIR}/${PARTIAL_FILE}"

  docker run --rm \
    --network none \
    --entrypoint tar \
    --volume "${BACKUP_VOLUME}:/data:ro" \
    --volume "${BACKUP_DIR}:/backup" \
    "${BACKUP_IMAGE_ID}" \
    --exclude='./cache' -czf "/backup/${PARTIAL_FILE}" -C /data .

  if [[ "${RESTART_AFTER_BACKUP}" == true ]]; then
    docker start autogpt >/dev/null
  fi

  BACKUP_SHA256="$(docker run --rm \
    --network none \
    --entrypoint sha256sum \
    --volume "${BACKUP_DIR}:/backup:ro" \
    "${BACKUP_IMAGE_ID}" \
    "/backup/${PARTIAL_FILE}" | awk '{print $1}')"
  if [[ ! "${BACKUP_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Backup checksum is not a valid SHA-256 digest" >&2
    exit 1
  fi
  printf '%s  %s\n' "${BACKUP_SHA256}" "${BACKUP_FILE}" \
    > "${BACKUP_DIR}/${CHECKSUM_PARTIAL}"
  mv "${BACKUP_DIR}/${PARTIAL_FILE}" "${BACKUP_DIR}/${BACKUP_FILE}"
  mv "${BACKUP_DIR}/${CHECKSUM_PARTIAL}" \
    "${BACKUP_DIR}/${CHECKSUM_FILE}"
  BACKUP_COMPLETE=true
  printf 'Backup written to %s with checksum %s\n' \
    "${BACKUP_DIR}/${BACKUP_FILE}" "${BACKUP_SHA256}"
  printf 'Image reference: %s\nImage digest: %s\nLocal image ID: %s\n' \
    "${BACKUP_IMAGE_REF}" "${BACKUP_IMAGE_DIGEST:-unavailable}" \
    "${BACKUP_IMAGE_ID}"
)
```

By default, the block restarts the unchanged installation as soon as the
archive completes, before calculating its checksum. The exit trap also attempts
a restart if a backup command fails. For an upgrade, export
`RESTART_AFTER_BACKUP=false` only for that one block and immediately `unset`
it afterward, whether the backup succeeds or fails. A successful upgrade backup
then leaves the appliance stopped at the cutover snapshot. With the default
setting, verify that it is running again:

```bash
docker inspect --format '{{.State.Status}}' autogpt
```

The atomic lock is a stopped helper container named `autogpt-backup-lock`, so it
serializes backups of `autogpt` even when callers choose different backup
directories. Catchable exits and signals remove it. After a shell is killed with
`SIGKILL` or the host crashes, first verify that no backup process is still
running; then remove only a stale lock with
`docker rm --volumes autogpt-backup-lock` before retrying. Inspect `BACKUP_DIR`
for stale `*.partial` files from the interrupted run and remove them only after
the same check. They can contain plaintext secrets and are not valid backups.

The block writes the checksum to `<archive>.sha256` and prints the container's
configured image reference, an immutable repository digest when available, and
the host-local image ID. Record an immutable tag or digest that resolves on the
restoring host, the environment file, and the Git commit beside the archive.
For a local-only build, preserve the image separately with an immutable registry
tag or `docker save`, and load it before restoring. The archive is plaintext and
contains user content, provider credentials, auth keys, and database passwords.
Encrypt it with an approved backup mechanism and remove unencrypted staging
copies.

The anonymous-volume guard also rejects a 64-character lowercase hexadecimal
volume name as a conservative fallback for Docker engines that omit the
anonymous-volume label. Use a conventional descriptive name for a manually
created volume.

## Restore into a new volume

Restore into a new named volume so the source remains recoverable. Set
`BACKUP_FILE` to the timestamped archive name and `RESTORE_IMAGE` to the
immutable tag or digest recorded with that backup. If the archive is not under
`./autogpt-backups`, also set `BACKUP_DIR` before running the block:

```bash
BACKUP_FILE=autogpt-data-YYYYMMDDTHHMMSSZ.tgz
RESTORE_IMAGE=significantgravitas/autogpt@sha256:RECORDED_DIGEST
# BACKUP_DIR=/absolute/path/to/autogpt-backups
```

```bash
(
  set -euo pipefail
  : "${BACKUP_FILE:?Set BACKUP_FILE to the timestamped archive filename}"
  : "${RESTORE_IMAGE:?Set RESTORE_IMAGE to the recorded immutable image}"
  BACKUP_DIR="${BACKUP_DIR:-${PWD}/autogpt-backups}"
  CHECKSUM_FILE="${BACKUP_FILE}.sha256"
  DEFAULT_RESTORE_VOLUME="autogpt-data-restored-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}${RANDOM}"
  RESTORE_VOLUME="${RESTORE_VOLUME:-${DEFAULT_RESTORE_VOLUME}}"
  RESTORE_OWNER_LABEL="org.agpt.restore.owner"
  RESTORE_OWNER="restore-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}${RANDOM}"
  RESTORE_CREATED=false
  RESTORE_COMPLETE=false

  if [[ "${BACKUP_FILE}" == */* ]]; then
    echo "BACKUP_FILE must be a filename within BACKUP_DIR" >&2
    exit 1
  fi
  if [[ ! -d "${BACKUP_DIR}" ]]; then
    echo "Backup directory does not exist: ${BACKUP_DIR}" >&2
    exit 1
  fi
  BACKUP_DIR="$(cd "${BACKUP_DIR}" && pwd -P)"
  if [[ ! -f "${BACKUP_DIR}/${BACKUP_FILE}" ]]; then
    echo "Backup archive does not exist: ${BACKUP_DIR}/${BACKUP_FILE}" >&2
    exit 1
  fi
  if [[ ! -f "${BACKUP_DIR}/${CHECKSUM_FILE}" ]]; then
    echo "Backup checksum does not exist: ${BACKUP_DIR}/${CHECKSUM_FILE}" >&2
    exit 1
  fi
  EXPECTED_SHA256="$(awk 'NR == 1 {print $1}' \
    "${BACKUP_DIR}/${CHECKSUM_FILE}")"
  if [[ ! "${EXPECTED_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Backup checksum is not a valid SHA-256 digest" >&2
    exit 1
  fi
  ACTUAL_SHA256="$(docker run --rm \
    --network none \
    --entrypoint sha256sum \
    --volume "${BACKUP_DIR}:/backup:ro" \
    "${RESTORE_IMAGE}" \
    "/backup/${BACKUP_FILE}" | awk '{print $1}')"
  if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "Backup checksum verification failed" >&2
    exit 1
  fi
  if docker volume inspect "${RESTORE_VOLUME}" >/dev/null 2>&1; then
    echo "Refusing to reuse existing volume: ${RESTORE_VOLUME}" >&2
    exit 1
  fi

  # Invoked by the EXIT and signal traps below.
  # shellcheck disable=SC2329
  finish_restore() {
    local exit_status="$1"
    local current_owner
    trap - EXIT HUP INT TERM
    if [[ "${RESTORE_CREATED}" == true && \
          "${RESTORE_COMPLETE}" != true ]]; then
      if ! current_owner="$(docker volume inspect \
        --format "{{ index .Labels \"${RESTORE_OWNER_LABEL}\" }}" \
        "${RESTORE_VOLUME}" 2>/dev/null)" || \
          [[ "${current_owner}" != "${RESTORE_OWNER}" ]]; then
        echo "Restore failed; refusing to remove a volume whose ownership cannot be verified: ${RESTORE_VOLUME}" >&2
        exit_status=1
      elif docker volume rm "${RESTORE_VOLUME}" >/dev/null; then
        echo "Restore failed; the partial restore volume was removed" >&2
      else
        echo "Restore failed; remove the partial volume after inspection: docker volume rm ${RESTORE_VOLUME}" >&2
        exit_status=1
      fi
    fi
    exit "${exit_status}"
  }

  trap 'finish_restore "$?"' EXIT
  trap 'finish_restore 129' HUP
  trap 'finish_restore 130' INT
  trap 'finish_restore 143' TERM
  docker volume create \
    --label "${RESTORE_OWNER_LABEL}=${RESTORE_OWNER}" \
    "${RESTORE_VOLUME}" >/dev/null
  if ! CREATED_OWNER="$(docker volume inspect \
    --format "{{ index .Labels \"${RESTORE_OWNER_LABEL}\" }}" \
    "${RESTORE_VOLUME}" 2>/dev/null)" || \
      [[ "${CREATED_OWNER}" != "${RESTORE_OWNER}" ]]; then
    echo "Refusing to populate a restore volume whose ownership cannot be verified: ${RESTORE_VOLUME}" >&2
    exit 1
  fi
  RESTORE_CREATED=true

  docker run --rm \
    --network none \
    --entrypoint tar \
    --volume "${RESTORE_VOLUME}:/data" \
    --volume "${BACKUP_DIR}:/backup:ro" \
    "${RESTORE_IMAGE}" \
    -xzf "/backup/${BACKUP_FILE}" -C /data

  RESTORE_COMPLETE=true
  printf 'Restored %s into volume %s\n' \
    "${BACKUP_FILE}" "${RESTORE_VOLUME}"
)
```

A per-run ownership label is checked after volume creation and again before
failure cleanup. If another run wins the same volume name, this restore refuses
to populate or remove that volume.

Use only an archive and checksum obtained through a trusted backup process. A
matching untrusted checksum detects accidental corruption but does not prove
who created the archive. Restore extraction has no network access, but `tar`
runs as root and preserves archive ownership and mode bits, including setuid
bits. Do not extract an archive from an untrusted source or assume network
isolation makes malicious archive content safe.

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
      test -s /data/postgres/postgresql.conf
      test -s /data/postgres/pg_hba.conf
      test -d /data/rabbitmq/mnesia
      test -d /data/valkey/17000
      test -d /data/valkey/17001
      test -d /data/valkey/17002
      test -d /data/falkordb
      test -d /data/workspaces
      test -d /data/home
      test -d /data/frontend-home
      quote="$(printf "\047")"
      setting="^[[:space:]]*listen_addresses([[:space:]]*=[[:space:]]*|[[:space:]]+)"
      hba_file_setting="^[[:space:]]*hba_file([[:space:]]*=[[:space:]]*|[[:space:]]+)"
      unsafe_setting="^[[:space:]]*(shared_preload_libraries|local_preload_libraries|"
      unsafe_setting="${unsafe_setting}session_preload_libraries|archive_mode|"
      unsafe_setting="${unsafe_setting}archive_command)"
      unsafe_setting="${unsafe_setting}([[:space:]]*=[[:space:]]*|[[:space:]]+)"
      include="^[[:space:]]*include(_if_exists|_dir)?([[:space:]]+|[[:space:]]*=)"
      end="[[:space:]]*(#.*)?$"
      set -- /data/postgres/postgresql.conf
      if test -f /data/postgres/postgresql.auto.conf; then
        set -- "$@" /data/postgres/postgresql.auto.conf
      fi
      active_listen="$(grep -hiE "${setting}" "$@" || true)"
      test "$(printf "%s\n" "${active_listen}" | grep -c .)" -eq 1
      printf "%s\n" "${active_listen}" | grep -Eiq "${setting}[[:space:]]*${quote}127[.]0[.]0[.]1${quote}${end}"
      if grep -Eiq "${include}" "$@" || \
          grep -Eiq "${hba_file_setting}" "$@" || \
          grep -Eiq "${unsafe_setting}" "$@"; then
        exit 1
      fi

      active_hba="$(grep -Ev "^[[:space:]]*(#|$)" /data/postgres/pg_hba.conf || true)"
      test "$(printf "%s\n" "${active_hba}" | grep -c .)" -eq 6
      hba_rule="^[[:space:]]*(local[[:space:]]+(all|replication)[[:space:]]+all[[:space:]]+peer|host[[:space:]]+(all|replication)[[:space:]]+all[[:space:]]+(127[.]0[.]0[.]1/32|::1/128)[[:space:]]+scram-sha-256)${end}"
      if printf "%s\n" "${active_hba}" | grep -Ev "${hba_rule}"; then
        exit 1
      fi
      for required_hba_rule in \
        "^[[:space:]]*local[[:space:]]+all[[:space:]]+all[[:space:]]+peer${end}" \
        "^[[:space:]]*host[[:space:]]+all[[:space:]]+all[[:space:]]+127[.]0[.]0[.]1/32[[:space:]]+scram-sha-256${end}" \
        "^[[:space:]]*host[[:space:]]+all[[:space:]]+all[[:space:]]+::1/128[[:space:]]+scram-sha-256${end}" \
        "^[[:space:]]*local[[:space:]]+replication[[:space:]]+all[[:space:]]+peer${end}" \
        "^[[:space:]]*host[[:space:]]+replication[[:space:]]+all[[:space:]]+127[.]0[.]0[.]1/32[[:space:]]+scram-sha-256${end}" \
        "^[[:space:]]*host[[:space:]]+replication[[:space:]]+all[[:space:]]+::1/128[[:space:]]+scram-sha-256${end}"
      do
        printf "%s\n" "${active_hba}" | grep -Eq "${required_hba_rule}"
      done
    '
)
```

The check also rejects PostgreSQL configuration that enables non-loopback
listening, loads external configuration fragments or libraries, redirects
`hba_file`, configures WAL archiving, or weakens the generated local `peer` and
loopback `scram-sha-256` access rules.
It does not prove that each database can start. A full
recovery rehearsal boots live schedules, stored credentials, and executors,
and some services fetch runtime data during startup. Perform it only on a
dedicated egress-filtered host or network after revoking or replacing
production provider and integration credentials. There is no generic appliance
switch that safely disables every possible outbound action. Retain both
volumes until the restore is accepted.

Never selectively mix service directories from different backups.

After accepting the restore, make sure no other container uses the `autogpt`
name or host port. Set `ENV_FILE` to the recorded environment file's absolute
host path. Set `PUBLISH_SPEC` to the original publish mapping when it was not
`127.0.0.1:3000:3000`. For a Docker Engine local-model installation, also set
`ADD_HOST_SPEC=host.docker.internal:host-gateway`; leave it empty otherwise.
Then launch the restored installation with the recorded image and volume:

```bash
ENV_FILE=/absolute/path/to/autogpt.env
RESTORE_VOLUME=autogpt-data-restored-YYYYMMDDTHHMMSSZ-PID-RANDOM
RESTORE_IMAGE=significantgravitas/autogpt@sha256:RECORDED_DIGEST
# PUBLISH_SPEC=127.0.0.1:3300:3000
# ADD_HOST_SPEC=host.docker.internal:host-gateway
```

```bash
(
  set -euo pipefail
  : "${ENV_FILE:?Set ENV_FILE to the recorded host environment-file path}"
  : "${RESTORE_VOLUME:?Set RESTORE_VOLUME to the restored volume name}"
  : "${RESTORE_IMAGE:?Set RESTORE_IMAGE to the recorded immutable image}"
  PUBLISH_SPEC="${PUBLISH_SPEC:-127.0.0.1:3000:3000}"
  ADD_HOST_SPEC="${ADD_HOST_SPEC:-}"
  NETWORK_DOCKER_ARGS=(--publish "${PUBLISH_SPEC}")
  if [[ -n "${ADD_HOST_SPEC}" ]]; then
    NETWORK_DOCKER_ARGS+=(--add-host "${ADD_HOST_SPEC}")
  fi

  [[ "${ENV_FILE}" = /* && -f "${ENV_FILE}" ]] || {
    printf 'ENV_FILE must be an existing absolute host path: %s\n' \
      "${ENV_FILE}" >&2
    exit 1
  }
  if ! docker volume inspect "${RESTORE_VOLUME}" >/dev/null 2>&1; then
    echo "Restored volume does not exist: ${RESTORE_VOLUME}" >&2
    exit 1
  fi
  if docker container inspect autogpt >/dev/null 2>&1; then
    echo "Refusing to replace an existing container named autogpt" >&2
    exit 1
  fi

  docker run --detach --name autogpt \
    --restart unless-stopped \
    --shm-size 2g \
    --ulimit nofile=65536:65536 \
    --log-driver json-file \
    --log-opt max-size=50m \
    --log-opt max-file=5 \
    "${NETWORK_DOCKER_ARGS[@]}" \
    --env-file "${ENV_FILE}" \
    --volume "${RESTORE_VOLUME}:/data" \
    "${RESTORE_IMAGE}"
)
```

## Upgrade and rollback

Before an upgrade:

1. Record the running image reference and image ID.
2. Pull or build the new image while the old appliance remains available.
3. Run `export RESTART_AFTER_BACKUP=false`, then run the Cold backup block once.
   Immediately run `unset RESTART_AFTER_BACKUP` afterward, whether the backup
   succeeds or fails. A successful backup leaves the appliance stopped; a
   failed backup restarts the unchanged installation.
4. Remove only the stopped container with `docker rm autogpt`.
5. Repeat the Quick start run command with the same environment file, named
   volume, publish mapping, `--add-host`, and other original launch options, but
   the new image reference.
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
docker logs --follow --tail 100 autogpt
```

`GET /healthz` checks nginx only; it is not proof that the whole appliance is
ready. The watchdog allows 600 seconds for initial full health; if that deadline
expires, it stops the container and `--restart unless-stopped` begins another
startup attempt.

| Symptom | What to check |
| --- | --- |
| The browser cannot connect after `docker run --rm "${IMAGE}"` | A bare run does not publish a port. Use the complete Quick start command. |
| Port `3300` opens but auth actions fail | Use `--publish 127.0.0.1:3300:3000` and set `AUTOGPT_PUBLIC_URL=http://localhost:3300`, then replace the container. |
| Signup shows **Email Not Allowed** | Inspect the API response or container logs to distinguish closed registration from an allowlist miss. Set `AUTH_ALLOW_NEW_ACCOUNTS=true` with an exact-address `AUTH_SIGNUP_ALLOWLIST=owner@example.com`, replace the container, create the intended accounts, and close signup again. |
| The container remains `starting` or becomes `unhealthy` | First boot can take several minutes. Run `autogpt-healthcheck` and inspect container logs for the first failed service. |
| The container is OOM-killed or repeatedly restarts during startup | The appliance uses about 5–6 GiB before workload headroom. On Docker Desktop, increase the VM memory allocation under **Settings → Resources**. |
| Startup refuses to continue after an interrupted migration | Follow the empty-install versus existing-install recovery procedure in [Quick start](#quick-start). The restart-looping container cannot reliably run `docker exec`; do not mark the migration applied or rolled back until you verify which database changes completed. |
| Startup rejects `DB_CONNECTION_LIMIT`, `DB_CONNECT_TIMEOUT`, or `DB_POOL_TIMEOUT` | Use an integer in the supported range: `1`–`5`, `1`–`600`, and `1`–`3600`, respectively. |
| Startup rejects legacy JWT secrets | Remove `JWT_VERIFY_KEY` and `SUPABASE_JWT_SECRET` for a fresh Better Auth installation. Set `AUTOGPT_ENABLE_LEGACY_AUTH=true` only for an intentional legacy-auth migration, and set both legacy variables to the same shared secret of at least 32 characters. |
| A run stays `RUNNING` without progress after a restart | The container stopped while the run was in flight. Its message was dropped and the row was not reconciled; start a new run. |
| Requests stall for minutes under concurrent runs | Inspect backend and PostgreSQL logs for connection-pool exhaustion. `DB_CONNECTION_LIMIT` cannot be raised above its default maximum of `5`; lowering `DB_POOL_TIMEOUT` makes pool exhaustion fail sooner but does not add capacity. Reduce concurrency or move to a distributed deployment when the fixed pools are insufficient. |
| AutoPilot returns a provider `401` | Configure the key for the selected transport. The default remote route needs `OPEN_ROUTER_API_KEY`; complete remote memory also needs `OPENAI_API_KEY`. |
| Local chat works but memory ingestion fails | Install the configured embedding model and confirm its `/v1/embeddings` endpoint works. If the server does not provide the default Qwen and `nomic-embed-text` slugs, set and install `GRAPHITI_LLM_MODEL`, `GRAPHITI_RERANKER_MODEL`, and `GRAPHITI_EMBEDDER_MODEL` explicitly. |
| Ollama cannot be reached | Keep the host-gateway option, ensure Ollama listens on an address Docker can reach, and test `/api/tags` from inside the container. |
| The container exits after a persistent health failure | The watchdog intentionally stops the appliance. Keep `--restart unless-stopped` so Docker can recover it. |
| Data appears missing after replacement | The new container is using another or anonymous `/data` volume. Inspect its mount and reattach the original named volume. |
| Restore launch reports the name `autogpt` or the host port is already in use | Inspect the leftover container and its `/data` mount. Free the intended host port, then run `docker rm autogpt` only after confirming the restored volume and launch settings are the intended replacement. |

## Known limitations

- One container is one failure, maintenance, scaling, and security boundary.
- The bundled frontend is compiled in local mode. Setting backend
  `BEHAVE_AS=cloud` does not create a supported multi-tenant hosted deployment.
- PostgreSQL, Valkey, RabbitMQ, FalkorDB, browser tooling, and the
  application compete for the same host resources.
- Valkey and FalkorDB have no appliance-level memory ceiling or eviction policy;
  their persisted working sets can grow until host memory or disk is exhausted.
  Monitor both resources and back up before capacity changes.
- Each backend role's Prisma pool has at most five PostgreSQL connections. The
  scheduler also has two SQLAlchemy job-store pools of three connections each.
  FalkorDB is configured for at most 25 queued queries and a 1-second query
  timeout. These fixed ceilings limit concurrency in the single-container
  distribution.
- All durable services share one volume and one backup schedule.
- There is no supported in-place conversion from the appliance to the
  multi-container or hosted deployment. Provision the destination separately
  and use agent export/import where supported; accounts, run history, schedules,
  credentials, and memory are not migrated automatically.
- Uploaded files are not scanned for malware. Unlike the hosted platform, this
  image bundles no antivirus daemon, so treat every upload as trusted input.
- Required email verification is unsupported.
- The prebuilt frontend cannot configure Google Picker public keys at runtime.
- Remote TLS termination is operator-supplied.
- The local `bash_exec` fallback depends on host support for Bubblewrap user and
  network namespaces. Do not make the entire appliance privileged to work
  around a host that disables them.
