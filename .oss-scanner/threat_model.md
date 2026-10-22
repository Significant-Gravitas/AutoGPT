# AutoGPT Platform threat model

## What AutoGPT is and how it is deployed

AutoGPT Platform (`autogpt_platform/` in this repository) is for building, running and sharing AI agents. Users
build agents as graphs of **blocks** (about 450 of them): LLM calls, web requests, file and media processing, code
execution, and integrations with dozens of third-party services. They can also ask **AutoPilot** (called "copilot"
in the code), the platform's own LLM agent, which builds and runs agents for them and has tools of its own: a
sandboxed shell, a headless browser, web fetch and search, file storage, MCP servers and chat-platform bots.

Both ways it is deployed are in scope:

- **The hosted service at https://platform.agpt.co**, run by the maintainers. It is multi-tenant and anyone can
  sign up. Users store third-party credentials (API keys and OAuth tokens), pay for usage with credits, publish
  agents to a public marketplace and work together in organizations (teams). Do not test against it: reproduce
  locally (see "How to exercise it").
- **Self-hosted**, with `autogpt_platform/docker-compose.yml` or the single-container image
  (`autogpt_platform/single-container/`), often for one person or a small team.

The main components:

- `autogpt_platform/backend/backend/`: the backend (Python, FastAPI, Prisma on PostgreSQL, Redis, RabbitMQ).
  - `api/`: the REST API (`rest_api.py`, routers in `api/features/*`), the WebSocket API (`ws_api.py`,
    `conn_manager.py`), the external API that API keys and OAuth apps use (`api/external/`), and AutoGPT's own
    OAuth 2.0 provider, "Sign in with AutoGPT" (`api/features/oauth.py`).
  - `data/`: the data layer, where most per-user and per-organization scoping happens.
  - `executor/` and `blocks/`: the engine that runs graphs, and the blocks. `util/request.py` is the HTTP client
    blocks are meant to use; it carries the SSRF protections.
  - `integrations/`: credential storage (`credentials_store.py`, `creds_manager.py`), third-party OAuth flows
    (`oauth/`) and inbound webhooks (`webhooks/`).
  - `copilot/`: AutoPilot. Its tools are in `copilot/tools/`. Code runs in an E2B sandbox
    (`tools/e2b_sandbox.py`) or, without E2B, in bubblewrap (`tools/sandbox.py`, `tools/bash_exec.py`). Its
    permission model is in `permissions.py`, `session_permissions.py` and `tools/capability_gates.py`; sharing is
    in `sharing/`, chat-platform bots in `bot/` and `../platform_linking/`.
  - `util/`: shared code, notably `encryption.py`, `cache.py`, `workspace.py`, `workspace_storage.py`,
    `cloud_storage.py`, `file.py`, `virus_scanner.py` and `secrets_guard.py`.
- `autogpt_platform/autogpt_libs/`: shared library. `auth/` verifies the JWTs the frontend's auth service issues;
  `api_key/` makes and checks API keys.
- `autogpt_platform/frontend/`: Next.js. It hosts the Better Auth service (`src/lib/auth/`, `src/app/api/auth/`),
  the session middleware (`src/middleware.ts`), a proxy to the backend (`src/app/api/proxy/[...path]/route.ts`)
  and admin impersonation (`src/lib/impersonation.ts`). It renders user- and model-generated content: markdown,
  code, and HTML previews in sandboxed iframes (`src/lib/iframe-sandbox-csp.ts`).

## Who the attacker is

In priority order:

1. **An unauthenticated client of the hosted service**: REST, WebSocket, external API, OAuth and auth endpoints,
   webhook ingress and file downloads.
2. **Any signed-up user** attacking other users, other organizations or the platform. Every user is untrusted, and
   the platform runs everyone's agents on shared infrastructure.
3. **Someone who shares something malicious**: an agent, marketplace listing, shared chat or file that a victim
   imports, runs or opens.
4. **Untrusted content an agent or AutoPilot processes**: web pages, emails, documents, API responses, webhook
   payloads and model output. Prompt injection counts only when it makes the platform cross a boundary (see "Out of
   scope").
5. **A third party sending webhooks**: the payload is untrusted until its signature is verified.
6. **A malicious web page** in a logged-in user's browser: CSRF, XSS, open redirects in the sign-in and OAuth
   flows.

Trusted: the platform's operators and admin-role users, the server's environment and configuration, and the
third-party services' own infrastructure.

## What we want reported

- **Authentication**: Better Auth sign-up, sign-in, password reset and email verification; session and JWT
  issuance and verification (`autogpt_libs/auth/`); API keys; the OAuth provider (PKCE, redirect URIs, scopes, code
  exchange, refresh tokens); admin impersonation.
- **Authorization**: user and organization isolation on every route, WebSocket subscription and executor path.
  That covers graphs, executions and their outputs, schedules, presets, the library, files and workspaces, chats
  and AutoPilot sessions, credentials, API keys, webhooks and notifications, plus admin-only routes and roles within
  an organization.
- **Credentials**: a user's stored third-party credentials must never be readable, usable or exfiltrable by anyone
  else. That includes through a graph, a shared or marketplace agent, AutoPilot, logs, error messages or exports.
  The same goes for platform-managed credentials and the server's own secrets.
- **Code execution**: user input must not run code on the backend or executor hosts. Known routes are blocks,
  templates, deserialization, file and media processing, and disabled blocks. AutoPilot's shell is meant to run
  only in E2B or bubblewrap. Escaping that sandbox to the host, to other sandboxes, to other users' files or, when
  networking is blocked, to the network is in scope.
- **SSRF**: any outbound request a user can direct must not reach private networks, link-local and cloud metadata
  addresses, or the platform's own services. That covers blocks, web fetch, MCP servers, webhooks, OAuth and MCP
  discovery, SMTP, RSS, and image and media URLs. Known bypasses include redirects, DNS rebinding, alternative IP
  encodings and IPv6 forms.
- **Money**: credits, budgets and rate limits, billing and Stripe webhooks. This covers anything that lets a user
  spend paid LLM or third-party usage at the platform's expense or someone else's.
- **Files and storage**: workspace uploads and downloads, signed URLs, marketplace media, path traversal,
  content-type and inline rendering, and getting past the virus scan.
- **Frontend**: stored or DOM XSS, especially from agent output, marketplace content, chats and markdown. Also
  escapes from the HTML-preview iframe sandbox, open redirects, CSRF, and tokens leaking through the backend proxy.
- **Denial of service** against shared services from one cheap request or agent run (see "How we rate severity").

## Past vulnerabilities worth looking for variants of

All are fixed and published at https://github.com/Significant-Gravitas/AutoGPT/security/advisories. Variants in
other blocks, endpoints and code paths are exactly what we want.

- **SSRF that bypasses the request wrapper**: via IPv6 (GHSA-4c8v-hwxc-2356), via IPv4-mapped IPv6 and CGNAT
  addresses (GHSA-8qc5-rhmg-r6r6) and via DNS rebinding (GHSA-wvjg-9879-3m7w); also cookies and protected headers
  forwarded across redirects (GHSA-ggcm-93qg-gfhp).
- **Blocks that make requests without the wrapper**: ReadRSSFeedBlock (GHSA-r55v-q5pc-j57f), SendDiscordFileBlock
  (GHSA-ggc4-4fmm-9hmc) and SendEmailBlock's SMTP host (GHSA-4jwj-6mg5-wrwf).
- **Blocks that should not be runnable**: running disabled blocks gave RCE (GHSA-r277-3xc5-c79v,
  GHSA-4crw-9p35-9x54), and executing a block directly skipped credit charges (GHSA-8pjg-mfqm-vrhr).
- **Cross-user access**: session hijacking through an IDOR (GHSA-q58p-v9r9-7gqj), node execution results leaking to
  other users over WebSockets (GHSA-958f-37vw-jx8f), and graph execution through the external API without
  authorization (GHSA-x77j-qg2x-fgg6).
- **Webhooks**: provider path confusion that skipped signature verification (GHSA-349p-3c3r-8mjr), binding
  another user's webhook to a preset (GHSA-4m2w-qfr5-9f3v), and an IDOR in the ping endpoint
  (GHSA-rq9m-xvc7-v9h6).
- **Secrets**: a hard-coded default JWT secret and Fernet key (GHSA-24q6-6h89-f9p7, GHSA-57mf-wqwq-6g6x), API keys
  logged in plain text (GHSA-rc89-6g7g-v5v7), and unsafe `pickle` deserialization of Redis cache entries
  (GHSA-rfg2-37xq-w4m9).
- **Frontend**: DOM XSS and an open redirect on the sign-up page (GHSA-j2cp-jg5q-38wj).
- **Resource exhaustion from one agent run or request**:
  - media blocks (GHSA-rg6v-m9x9-7wf9, GHSA-g26x-xwc5-7p44, GHSA-267x-8jx3-gg6w)
  - screenshot and file blocks (GHSA-7g34-7fvq-xxq6, GHSA-9fr4-9jj9-mhh6)
  - regular expressions, text templating and text chunking (GHSA-m2wr-7m3r-p52c, GHSA-pppq-xx2w-7jpq,
    GHSA-ppw9-h7rv-gwq9, GHSA-955p-gpfx-r66j)
  - unauthenticated disk exhaustion (GHSA-374w-2pxq-c9jp)

## Out of scope

- **`classic/`** (the original AutoGPT agent, Forge and the benchmark) is unsupported; see `SECURITY.md`.
- **Development defaults documented as such**: the RabbitMQ credentials and database passwords in the
  docker-compose files, in CI and in `.env.default`, and the dev-only VAPID keypair there. A default that the
  application *accepts in a production configuration* is in scope; the JWT-secret and Fernet-key advisories above
  are examples.
- **Prompt injection on its own**, meaning a model that says something, or acts within the permissions its user
  already gave the session. It is in scope when injected content makes the platform cross a boundary listed above:
  using credentials or data the session was not granted, reaching another user's data, escaping a sandbox,
  getting past a permission or approval the user set, or exfiltrating platform secrets.
- **A user attacking only themselves**: self-XSS, running code in their own sandbox, spending their own credits.
- **Local-development behavior.** Deployments run `APP_ENV=dev` or `prod` and `BEHAVE_AS=cloud`. Anything that
  happens only under `APP_ENV=local` or `BEHAVE_AS=local` is out of scope: the `/docs` page, the test-data routes,
  `FORCE_ALL_FLAGS`, the Teams bot's unverified mode, and the paid features and limits a local install waives.
  `APP_ENV` is `local` in this image so that flags can be forced on.
- **Vulnerabilities in third-party dependencies or services**, unless the way AutoGPT uses them creates the
  problem.
- **Rate-limit and anti-abuse gaps** with no security or cost impact.
- **Other paths**: `docs/` (except content the backend serves to AutoPilot), tests, and
  `autogpt_platform/analytics/`.

## Which code matters most

Backend paths are relative to `autogpt_platform/backend/` (the Python package is `backend`), frontend paths to
`autogpt_platform/frontend/`.

- **Highest**:
  - `backend/api/` (including `api/external/` and `api/features/oauth.py`), `autogpt_libs/auth/`,
    `autogpt_libs/api_key/`
  - `backend/data/`: the queries that scope by user and organization
  - `backend/integrations/`: credentials, OAuth and webhooks
  - `backend/copilot/tools/` and the sandboxes, plus `backend/copilot/permissions.py` and
    `session_permissions.py`
  - `backend/util/request.py`, and every block that makes network calls or touches files
  - `backend/executor/`
  - frontend `src/lib/auth/`, `src/middleware.ts` and `src/app/api/`
- **Medium**:
  - the rest of `backend/blocks/` and `backend/copilot/`
  - `backend/notifications/`, `backend/platform_linking/`
  - the marketplace (`api/features/store/`), billing (`api/features/billing/`, `data/credit.py`) and organizations
    (`api/features/orgs/`)
  - frontend components that render user or model content
- **Lower**:
  - `autogpt_platform/installer/`, `autogpt_platform/single-container/`, `autogpt_platform/swap_proxy/`,
    `autogpt_platform/cloudflare_worker.js`, `autogpt_platform/graph_templates/`

## How to exercise it

The image has no network access; everything below works offline. The backend is installed in
`autogpt_platform/backend/.venv` (run it through `poetry run`), the frontend in
`autogpt_platform/frontend/node_modules`, with a production build in `.next/`.

**Services.** `autogpt-services start` starts:

- PostgreSQL 15 with pgvector on `:5432` (user `postgres`, trust auth); the database is already migrated
- a 3-shard Redis cluster (Valkey) on `:17000`-`:17002`
- RabbitMQ on `:5672`
- ClamAV on `:3310`, which scans every upload. Its only signature is the EICAR test file's, so use that file to
  show a scan being bypassed.

`autogpt-services status` and `autogpt-services stop` do what they say. Logs are in `/var/log/autogpt-services`.

**Backend tests** use pytest, with `*_test.py` files next to the code they test:

```
autogpt-services start
. /etc/autogpt-scanner/test.env      # the environment backend CI uses
cd /src/autogpt_platform/backend
poetry run pytest -q backend/api/features/library
```

- Tests that need the Internet fail or skip offline, as expected: LLM providers, third-party APIs, E2B, cloud
  storage, and the cases in `backend/util/request_test.py` that resolve real hostnames.
- The whole suite is slow on two CPUs; CI splits it four ways (`.github/scripts/backend_test_shard.py`). Run the
  directories you need.
- The route tests (`backend/api/features/**/*_test.py`) show how to call a route as a given user with FastAPI's
  `TestClient`. That is the quickest way to demonstrate an authorization bug.

**The platform**, to reproduce through real HTTP. Start only the backend services you need. The four below take
about 5 GB; all of the backend's services at once (`poetry run app`) take more memory than this machine has.

```
autogpt-services start
. /etc/autogpt-scanner/stack.env
cd /src/autogpt_platform/backend
(poetry run rest > /tmp/rest.log 2>&1 &)           # REST API :8006
(poetry run db > /tmp/db.log 2>&1 &)               # database manager :8005, which the other services call
(poetry run executor > /tmp/executor.log 2>&1 &)   # runs agents
cd /src/autogpt_platform/frontend
(pnpm start > /tmp/frontend.log 2>&1 &)           # Next.js and Better Auth :3000
```

- The other entry points are `ws` (WebSocket API, `:8001`), `scheduler`, `notification` and `copilot-executor`.
- `stack.env` runs the backend as the hosted service does: `BEHAVE_AS=cloud`, with credits on.
- **Accounts** are made through Better Auth: `POST http://localhost:3000/api/auth/sign-up/email` with
  `{"email": ..., "password": ..., "name": ...}` (passwords are at least 12 characters), an
  `Origin: http://localhost:3000` header, and a cookie jar. Email verification is off, and sign-up also creates the
  user's backend record.
- **Tokens**: `GET http://localhost:3000/api/auth/token` with that cookie returns `{"token": ...}`, a JWT valid
  for an hour. The backend takes it as `Authorization: Bearer <jwt>`.
- **Admins**: `psql -h localhost -U postgres -c "UPDATE \"UserAuthIdentity\" SET role = 'admin' WHERE email = '...'"`,
  then sign in again (`POST http://localhost:3000/api/auth/sign-in/email` with the email and password) and fetch a
  token from the new session; the old session keeps the old role. An admin can give a user credits with
  `POST /api/credits/admin/add_credits`.
- **Feature flags** come from LaunchDarkly, which isn't configured here, so every flag is off.
  `FORCE_FLAG_<KEY>=true` in the backend's environment turns one on (the flag key upper-cased, with `-` as `_`), and
  `FORCE_ALL_FLAGS=true` turns them all on. Code behind a flag is in scope: the hosted service turns flags on for
  some or all users.
- `next start` warns that it doesn't work with standalone output; it serves everything anyway.
- Anything that needs an API key logs errors and fails offline (LLM calls, embeddings, third-party integrations).

**Frontend tests** use Vitest with MSW, which works offline. Most test files are in `__tests__` folders next to the
code they test:

```
cd /src/autogpt_platform/frontend
pnpm exec vitest run src/lib/auth
```

- The whole suite takes over 30 minutes on two CPUs, so run the folders you need; `src/lib/auth` takes 40 seconds.
- Playwright browsers are not installed.

## How we rate severity

- **Critical**:
  - remote code execution on the backend or executor hosts, by anyone with an account or none
  - authentication bypass or takeover of arbitrary accounts
  - reading or using another user's stored credentials, or the platform's secrets and cloud credentials (for
    example through SSRF to a metadata endpoint)
  - access to any user's or organization's private data at will
- **High**:
  - escaping AutoPilot's sandbox to the host
  - SSRF that returns responses from internal addresses
  - IDOR that reads or changes another user's or organization's private data, case by case
  - privilege escalation to admin, or to a higher role within an organization
  - stored XSS that runs in another user's session
  - a webhook that triggers runs in someone else's account without a valid signature
  - running disabled or admin-only blocks
  - spending the platform's money (LLM or third-party usage) past credit or budget limits
  - one cheap request or agent run that takes a shared service (API, executor, database) down for everyone
- **Medium**:
  - blind SSRF
  - CSRF that changes state
  - reflected or DOM XSS that needs a click
  - open redirects in the sign-in and OAuth flows (high if they leak a token or code)
  - disclosure of other users' non-secret metadata, such as email addresses
- **Low**:
  - denial of service that needs sustained traffic, or that only hurts the attacker's own runs
  - missing hardening headers and verbose errors
  - problems that need unusual, non-default configuration

## How reports should look

- Name the attacker from "Who the attacker is" that the exploit assumes: no account, any account, a member of the
  same organization, a webhook sender, or a victim who must click, open or run something.
- Reproduce against an unmodified checkout, set up as `stack.env` or `test.env` does, and say which other settings
  the exploit needs, if any.
- One report per root cause. If the same missing check affects several routes or blocks, list them all in one
  report.
- The reproducer should be a pytest test (a `*_test.py` file next to the code), or a short script or `curl`
  sequence against the local platform started as above, with the expected and actual result.
- Patches should be minimal and against the `dev` branch, with a regression test. Backend code is formatted with
  `poetry run format`, frontend code with `pnpm format`.
