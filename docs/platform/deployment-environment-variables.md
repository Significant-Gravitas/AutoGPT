# Deployment Environment Variables

This page is for running the Docker Compose stack from
[Manual Setup](getting-started.md#manual-setup) on a server that people reach
from other machines. It lists only the settings you must change for that, and
why. Every setting, with its default and what it does, is described where it
is set:

- [`autogpt_platform/.env.default`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/.env.default)
- [`autogpt_platform/backend/.env.default`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/backend/.env.default)
- [`autogpt_platform/frontend/.env.default`](https://github.com/Significant-Gravitas/AutoGPT/blob/master/autogpt_platform/frontend/.env.default)

To put the single-container image on a server, see
[Run AutoGPT in One Docker Container](single-container.md) or the
[installer](installer.md) instead: they take one public URL and work out the
rest.

## Where settings live

`make init-env` (run in `autogpt_platform/`) copies each `.env.default` to
`.env` and fills in the secrets they leave blank. Make your changes in the
`.env` files. Never edit `.env.default`: it is public, and the next update
overwrites it.

| File | Read by |
|------|---------|
| `autogpt_platform/.env` | Docker Compose itself. It fills the `${...}` values in the compose files: the database password, and the two frontend URLs below. |
| `autogpt_platform/backend/.env` | Every backend service, and the Redis, RabbitMQ and FalkorDB containers. |
| `autogpt_platform/frontend/.env` | The frontend, which also runs the login service (Better Auth). |

Inside Docker, the compose files set some values themselves, and those win
over `backend/.env` and `frontend/.env`: the hostnames of the other services,
`DATABASE_URL`, `DIRECT_URL`, `JWT_JWKS_URL` and `FRONTEND_BASE_URL` for the
backend, and `DATABASE_URL`, `AGPT_SERVER_URL`, `AGPT_WS_SERVER_URL` and
`NEXT_PUBLIC_FRONTEND_BASE_URL` for the frontend. Changing one of those in
`backend/.env` or `frontend/.env` does nothing under Docker. The two URLs among
them are taken from `autogpt_platform/.env`, which is why they go there below.

## 1. Generate your secrets

```bash
cd AutoGPT/autogpt_platform
make init-env
```

This generates `ENCRYPTION_KEY`, `UNSUBSCRIBE_SECRET_KEY`, `INTERNAL_API_KEY`,
`BETTER_AUTH_SECRET`, the Web Push key pair (`VAPID_PRIVATE_KEY` and
`VAPID_PUBLIC_KEY`), and the passwords of the database, RabbitMQ, Redis and
FalkorDB, for your install only. There is nothing to generate by hand. Most of
them are required: without `ENCRYPTION_KEY` the backend refuses to start,
without `POSTGRES_PASSWORD` Docker Compose refuses to start, without the
FalkorDB password its container exits, without the RabbitMQ password the
backend cannot log in to it, and without `BETTER_AUTH_SECRET` sign-in fails.
The rest can be empty: `INTERNAL_API_KEY` is then derived from
`ENCRYPTION_KEY`, and without `REDIS_PASSWORD` or `UNSUBSCRIBE_SECRET_KEY` the
backend logs a warning and runs with Redis open to the other containers and
email unsubscribe links unsigned.

The database password is in three places: `POSTGRES_PASSWORD` in
`autogpt_platform/.env`, which Docker uses for the database and for every
connection string, and its copies `DB_PASS` in `backend/.env` and the password
in `DATABASE_URL` in `frontend/.env`, which only matter outside Docker. To
change it, follow
[Upgrading: service passwords are generated per install](getting-started.md#upgrading-service-passwords-are-generated-per-install).

Set one thing yourself: `VAPID_CLAIM_EMAIL` in `backend/.env`, to `mailto:`
an address you read. Push services use it to contact you about this server's
notifications. Leave `NEXT_PUBLIC_VAPID_PUBLIC_KEY` in `frontend/.env` blank:
the frontend fetches the public key from the backend.

## 2. Set your public URLs

Every default points at `localhost`, which only works in a browser on the
server itself. From any other machine, people can still sign up and log in,
but the browser then looks for the backend on its own machine: live updates
never connect, and the API refuses the calls the browser makes to it directly,
such as chatting with Otto. Set these to where people reach the frontend and
the backend.

The examples use `https://autogpt.example.com` for the frontend and
`https://api.autogpt.example.com` for the backend, behind a reverse proxy (see
[step 3](#3-put-it-behind-https)). Without one, use the server's address and the
published ports instead: `http://<server>:3000` for the frontend,
`http://<server>:8006` for the backend (`http://<server>:8006/api` in
`NEXT_PUBLIC_AGPT_SERVER_URL`) and `ws://<server>:8001/ws` for the WebSocket.

| File | Setting | Example | What it is for |
|------|---------|---------|----------------|
| `autogpt_platform/.env` | `FRONTEND_BASE_URL` | `https://autogpt.example.com` | Links the backend sends people: OAuth redirects, emails, Stripe |
| `autogpt_platform/.env` | `NEXT_PUBLIC_FRONTEND_BASE_URL` | `https://autogpt.example.com` | The frontend's own address |
| `frontend/.env` | `BETTER_AUTH_URL` | `https://autogpt.example.com` | The login service's own address, used in password-reset and verification links and social sign-in callbacks |
| `frontend/.env` | `BETTER_AUTH_INTERNAL_URL` | `http://localhost:3000` | Where the frontend's server reaches its own login service. Always `http://localhost:3000` under Docker |
| `frontend/.env` | `NEXT_PUBLIC_AGPT_SERVER_URL` | `https://api.autogpt.example.com/api` | The backend API, as the browser reaches it |
| `frontend/.env` | `NEXT_PUBLIC_AGPT_WS_SERVER_URL` | `wss://api.autogpt.example.com/ws` | The WebSocket server, as the browser reaches it |
| `backend/.env` | `PLATFORM_BASE_URL` | `https://api.autogpt.example.com` | Webhook and MCP URLs the backend hands to other services |
| `backend/.env` | `BACKEND_CORS_ALLOW_ORIGINS` | `["https://autogpt.example.com"]` | Origins the API and WebSocket server accept calls from (a JSON list; the default only allows `http://localhost:3000`) |

Some of these lines are not in the files yet, so add them:
`FRONTEND_BASE_URL` and `NEXT_PUBLIC_FRONTEND_BASE_URL` in
`autogpt_platform/.env`, `BACKEND_CORS_ALLOW_ORIGINS` in `backend/.env`, and
`BETTER_AUTH_INTERNAL_URL` in `frontend/.env`. Leave out the trailing `/`.

Without `BETTER_AUTH_INTERNAL_URL`, the frontend's server calls
`BETTER_AUTH_URL` to fetch a signed-in visitor's token. That works only if the
frontend container can resolve your public hostname, reach it, and trust its
certificate. With a name only your network's clients know about, or a
self-signed or internal certificate, it cannot: sign-up then fails, and the
sign-up page stays where it is without an error.

Give `FRONTEND_BASE_URL`, `NEXT_PUBLIC_FRONTEND_BASE_URL` and
`BETTER_AUTH_URL` the same value. The backend sends password-reset and
verification emails only for links on `FRONTEND_BASE_URL`; to allow other
origins as well, list them in `TRUSTED_FRONTEND_ORIGINS` in `backend/.env`.

The `NEXT_PUBLIC_*` values are built into the frontend, so rebuild it after
changing one:

```bash
docker compose up -d --build
```

Leave `JWT_JWKS_URL` alone under Docker: the backend fetches the login keys
from the frontend over the Docker network (`http://frontend:3000`), and the
compose file sets that for you. If you run the backend and frontend on
different machines, read
[Auth transport security](getting-started.md#auth-transport-security-jwks-over-untrusted-networks)
first.

## 3. Put it behind HTTPS

Docker publishes three ports to the network: the frontend on `3000`, the API
on `8006` and the WebSocket server on `8001`. They speak plain HTTP, and
logins and session cookies go over them. Anything other people use should be
behind a reverse proxy (nginx, Caddy, Traefik) that terminates TLS. With the
example names:

| Public address | Proxy to |
|----------------|----------|
| `https://autogpt.example.com` | `http://127.0.0.1:3000` |
| `https://api.autogpt.example.com/ws` | `http://127.0.0.1:8001` (WebSocket upgrade on) |
| everything else on `https://api.autogpt.example.com` | `http://127.0.0.1:8006` |

For example, in a `Caddyfile`:

```text
autogpt.example.com {
	reverse_proxy 127.0.0.1:3000
}

api.autogpt.example.com {
	reverse_proxy /ws* 127.0.0.1:8001
	reverse_proxy 127.0.0.1:8006
}
```

Use two hostnames, not two paths on one: the frontend serves its own routes
under `/api` (logins go to `/api/auth`), so it cannot share a host with the
backend's `/api`.

Then close `3000`, `8001` and `8006` to everything but the proxy, with your
firewall or a cloud security group. The other ports the stack publishes
(Postgres, Redis, RabbitMQ, FalkorDB and ClamAV) are bound to `127.0.0.1`
only.

Set `TRUSTED_PROXIES` in both `backend/.env` and `frontend/.env` to your
proxies' addresses, as a comma-separated list of IP addresses or CIDR ranges,
for example `10.0.0.0/8,172.16.0.0/12`. Each takes the visitor to be the first
`X-Forwarded-For` address from the right that the list does not include, and
the backend reads the header only from a listed peer, so the proxies may
append to the header or replace it. The backend uses it for the client address
of every request, including the External API's (`/api/v2`) per-IP rate
limits. The frontend uses it for its per-IP limits on sign-in and
password-reset attempts, the per-visitor limit on store search, and the
sign-in history and IP blocks. Without it, sign-in and password-reset attempts
are limited only per email address, all visitors share one store-search limit,
and sign-ins record no IP address. An entry that is not an IP address or CIDR
range stops the service from starting.

`TRUSTED_PROXY_COUNT` and `FORWARDED_ALLOW_IPS` still work in `backend/.env`
while `TRUSTED_PROXIES` is unset, but are deprecated. The frontend no longer
reads `TRUST_X_FORWARDED_FOR`.

## 4. Decide who can sign up

Anyone who can reach the frontend can create an account. To limit that, set
one of these in `frontend/.env`:

- `AUTH_SIGNUP_ALLOWLIST`: a comma-separated list of the addresses and
  `@domains` allowed to sign up, for example `@example.com,me@example.org`.
- `AUTH_ALLOW_NEW_ACCOUNTS=false`: no new accounts at all, once yours exists.

Sign-up does not check email addresses unless you set
`AUTH_REQUIRE_EMAIL_VERIFICATION=true`. That, and password reset, need the
backend's mailer: set `POSTMARK_SERVER_API_TOKEN` and `POSTMARK_SENDER_EMAIL`
in `backend/.env`. There is no SMTP setting for login emails.

## 5. Register OAuth callback URLs

Integrations (GitHub, Google and the others in `backend/.env`) send people
back to the frontend. Register this callback with each provider:

```text
https://autogpt.example.com/auth/integrations/oauth_callback
```

That is `FRONTEND_BASE_URL` followed by `/auth/integrations/oauth_callback`,
with no internal port. The client IDs and secrets go in `backend/.env`.

Signing in with Google, GitHub or Discord is set up separately, with the
`AUTH_*_CLIENT_ID` and `AUTH_*_CLIENT_SECRET` pairs in `frontend/.env`. Their
callback is `BETTER_AUTH_URL` followed by `/api/auth/callback/<provider>`, for
example `https://autogpt.example.com/api/auth/callback/google`.

## Using your own database or Redis

The compose files point every backend service at the bundled `db`, `redis-0`
and `rabbitmq` containers, and set `DB_HOST`, `REDIS_HOST`, `REDIS_PORT`
(`17000`), `RABBITMQ_HOST`, `DATABASE_URL` and `DIRECT_URL` in their
`environment:` sections. Those override `backend/.env`, so pointing the stack
at a managed service takes a `docker-compose.override.yml` that sets them per
service, not an `.env` change. Override `REDIS_PORT` along with `REDIS_HOST`,
or the backend still connects to port `17000`.
`DIRECT_URL` is the connection that database migrations use; give it a direct
connection, not one through a pooler such as PgBouncer. The backend expects a
Redis Cluster, not a single Redis server.

For Redis there is a shorter way: the backend reads `REDIS_CLUSTER_HOST` and
`REDIS_CLUSTER_PORT` before `REDIS_HOST` and `REDIS_PORT`, and the compose
files do not set them, so setting those two in `backend/.env` is enough.

## Checklist

- [ ] `make init-env` in `autogpt_platform/`
- [ ] A real `mailto:` `VAPID_CLAIM_EMAIL` in `backend/.env`
- [ ] `FRONTEND_BASE_URL` and `NEXT_PUBLIC_FRONTEND_BASE_URL` added to `autogpt_platform/.env`
- [ ] `BETTER_AUTH_URL`, `BETTER_AUTH_INTERNAL_URL`, `NEXT_PUBLIC_AGPT_SERVER_URL` and `NEXT_PUBLIC_AGPT_WS_SERVER_URL` in `frontend/.env`
- [ ] `PLATFORM_BASE_URL` and `BACKEND_CORS_ALLOW_ORIGINS` (added) in `backend/.env`
- [ ] A TLS reverse proxy in front of `3000`, `8001` and `8006`, those ports firewalled, and `TRUSTED_PROXIES` set in `backend/.env` and `frontend/.env`
- [ ] Sign-ups limited, if the server is public
- [ ] `docker compose up -d --build`
- [ ] From another machine: sign up, log in, and check the browser console shows no `localhost` connection errors

## Related documentation

- [Getting Started](getting-started.md): the full setup, upgrades and the Makefile
- [Advanced Setup](advanced_setup.md)
- [OAuth & SSO](integrating/oauth-guide.md)
