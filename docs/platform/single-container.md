# Single-container AutoGPT Platform (Experimental)

!!! danger
    This distribution is experimental. It is intended for local evaluation and
    single-host self-hosting, not production. Do not publish it to Docker Hub
    until the license, image-content, vulnerability, fresh-install, upgrade, and
    rollback gates in this document have been completed.

The single-container image packages the AutoGPT Platform web application and
its supporting processes behind one public port. Docker Compose manages exactly
one container; a process supervisor inside that container manages the frontend,
backend roles, PostgreSQL, a three-node Redis cluster, RabbitMQ, ClamAV, and
FalkorDB. The Redis-compatible cluster is implemented with Valkey. Only port
`3000` is published. The internal backend proxy routes are `/_agpt/api` and
`/_agpt/ws`.

This is a distribution format, not a replacement for the existing
multi-container development Compose stack.

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
  up --build -d
```

Follow startup and wait for the container to become healthy:

```bash
docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  logs -f autogpt
```

Open `http://localhost:3000`. First startup is slower than later starts because
the volume must be initialized, database migrations must run, and service data
may need to be downloaded or generated.

The built image has a complete default command: `docker run autogpt` starts the
whole appliance without Compose. Publish the port and mount `/data` for normal
use:

```bash
docker build \
  --target single-container \
  -f autogpt_platform/backend/Dockerfile \
  -t autogpt .

docker run --name autogpt \
  --restart unless-stopped \
  --stop-timeout 360 \
  --shm-size 2g \
  --ulimit nofile=65536:65536 \
  --log-opt max-size=50m \
  --log-opt max-file=5 \
  -p 127.0.0.1:3000:3000 \
  -v autogpt-platform-data:/data \
  autogpt
```

If the host-facing port is changed, pass the exact browser origin as well. For
example:

```bash
docker run --name autogpt \
  --restart unless-stopped \
  --stop-timeout 360 \
  --shm-size 2g \
  --ulimit nofile=65536:65536 \
  --log-opt max-size=50m \
  --log-opt max-file=5 \
  -p 127.0.0.1:3300:3000 \
  -e AUTOGPT_PUBLIC_URL=http://localhost:3300 \
  -v autogpt-platform-data:/data \
  autogpt
```

The image defaults to `http://localhost:3000`; Docker cannot infer a different
host-facing port from inside the container. `AUTOGPT_PUBLIC_URL` configures the
auth issuer, callbacks, cookies, and generated links. The appliance proxy also
rewrites private frontend redirects to that validated public origin.

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
  up -d --no-build
```

Record the registry manifest digest from
`docker buildx imagetools inspect IMAGE:TAG` in the deployment record. A local
image ID from `docker compose images` is not the multi-platform registry
manifest digest, and a mutable tag alone is not a sufficient rollback record.

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
over plain HTTP. The proxy must:

- preserve the original `Host` and forwarded protocol;
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

Remove any temporary signup allowance after the account exists. Keep host-level
access to `docker compose exec` limited to trusted administrators.

## LLM providers and local inference

The image starts without a provider key, but AI-backed features need an
appropriate provider. Set only the credentials you use in the local `.env`:

```dotenv
OPEN_ROUTER_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GROQ_API_KEY=
```

For an OpenAI-compatible server such as Ollama running on the Docker host:

```dotenv
CHAT_USE_LOCAL=true
CHAT_BASE_URL=http://host.docker.internal:11434/v1
CHAT_API_KEY=ollama
CHAT_FAST_STANDARD_MODEL=hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M
```

The fast-standard model is required for local inference. The title,
simulation, and fast-advanced models inherit it unless explicitly overridden.
The thinking-standard and thinking-advanced fields do not inherit it; local
transport currently downgrades extended-thinking requests to the baseline
path. Keep the extended-thinking UI disabled unless separately validated with
compatible model overrides. See the [local LLM guide](copilot-local-llm.md)
for model sizing and advanced-tier options. On Linux, Compose adds
`host.docker.internal` to the container, but still verify the selected
inference server is listening on an address the container can reach.

The Compose service also passes through additional variables from
`autogpt_platform/single-container/.env`, including block credentials and
provider-specific settings not listed in the example. Always keep the
documented `--env-file` option on `docker compose` commands: values listed
explicitly in the Compose model are interpolated from that file before the
service-level pass-through is applied. The example contains a guard variable
that makes Compose fail model resolution if the option is omitted; this avoids
silently reversing account or origin policy through Compose precedence.

Keep this file mode `0600`. Provider and OAuth credentials passed as container
environment variables are visible to Docker-daemon administrators and through
privileged container inspection. The appliance-generated database, encryption,
and session secrets are instead kept in `/data/config/runtime.env` at mode
`0600`, but anyone with Docker or volume access can still read them.

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
`GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`. Their shared callback is:

```text
https://agents.example.com/auth/integrations/oauth_callback
```

Social login providers use Better Auth callbacks below `/api/auth/callback/`.
Create provider applications only after `AUTOGPT_PUBLIC_URL` and TLS are final,
and copy callback URLs exactly from the provider configuration screen.

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

All durable state lives in the named volume mounted at `/data`. Its default
name is `autogpt-platform-data`, configurable with `AUTOGPT_DATA_VOLUME`. It
contains application databases, queue state, workspaces, service data, and the
generated runtime secrets that protect sessions and stored credentials.

Consequences:

- `docker compose down` preserves data; `docker compose down -v` deletes it.
- Losing the generated secret file can make encrypted credentials unusable and
  invalidate every session, even if a database copy survives.
- A volume backup contains user content, provider credentials, auth keys, and
  database passwords. Encrypt it at rest and tightly restrict access.
- Do not mount one volume into two running AutoGPT containers.

### Cold backup

A stopped-volume archive is the supported first backup mechanism for this
experimental image. It is intentionally simple and restores all coupled
services to the same point in time.

```bash
umask 077
export AUTOGPT_IMAGE=YOUR_DOCKERHUB_ACCOUNT/autogpt-platform
export AUTOGPT_TAG=experimental-YYYYMMDD-SHORT_SHA
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
  "${AUTOGPT_IMAGE}:${AUTOGPT_TAG}" \
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
  "${AUTOGPT_IMAGE}:${AUTOGPT_TAG}" \
  -xzf /backup/autogpt-platform-data.tgz -C /data

AUTOGPT_DATA_VOLUME="${RESTORED_VOLUME}" \
docker compose \
  --env-file autogpt_platform/single-container/.env \
  -f autogpt_platform/docker-compose.single-container.yml \
  up -d --no-build
```

Verify the restore before changing the deployment's normal volume setting.

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
  up -d --no-build
```

Do not run an older image against a volume after newer migrations have modified
it. A rollback means selecting the prior image tag **and restoring the matching
pre-upgrade archive into a new volume**. This is why the backup and image digest
are part of the same deployment record.

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
