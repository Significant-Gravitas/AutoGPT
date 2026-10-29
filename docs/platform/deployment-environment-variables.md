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
`BETTER_AUTH_SECRET`, and the passwords of the database, RabbitMQ, Redis and
FalkorDB, for your install only. There is nothing to generate by hand, and the
stack does not start without them.

The database password is in three places: `POSTGRES_PASSWORD` in
`autogpt_platform/.env`, which Docker uses for the database and for every
connection string, and its copies `DB_PASS` in `backend/.env` and the password
in `DATABASE_URL` in `frontend/.env`, which only matter outside Docker. To
change it, follow
[Upgrading: service passwords are generated per install](getting-started.md#upgrading-service-passwords-are-generated-per-install).

## 2. Replace the shared push keys

`make init-env` does not generate the Web Push keys. The ones in
`backend/.env.default` are a published development pair, so make your own
before anyone else uses the server. With the stack running
(`docker compose up -d`), this prints a new pair from inside the backend
container:

```bash
docker compose exec -T rest_server python - <<'EOF'
import base64
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from py_vapid import Vapid
v = Vapid()
v.generate_keys()
private = v.private_key.private_numbers().private_value.to_bytes(32, "big")
public = v.public_key.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
print("VAPID_PRIVATE_KEY=" + base64.urlsafe_b64encode(private).rstrip(b"=").decode())
print("VAPID_PUBLIC_KEY=" + base64.urlsafe_b64encode(public).rstrip(b"=").decode())
EOF
```

Then set:

| File | Setting | Value |
|------|---------|-------|
| `backend/.env` | `VAPID_PRIVATE_KEY` | Your private key |
| `backend/.env` | `VAPID_PUBLIC_KEY` | Your public key |
| `backend/.env` | `VAPID_CLAIM_EMAIL` | `mailto:` an address you read |
| `frontend/.env` | `NEXT_PUBLIC_VAPID_PUBLIC_KEY` | The same public key |

## 3. Set your public URLs

Every default points at `localhost`, which only works in a browser on the
server itself. From any other machine, people can still sign up and log in,
but the browser then looks for the backend on its own machine: live updates
never connect, and the API refuses the calls the browser makes to it directly,
such as chatting with Otto. Set these to where people reach the frontend and
the backend.

The examples use `https://autogpt.example.com` for the frontend and
`https://api.autogpt.example.com` for the backend, behind a reverse proxy (see
[step 4](#4-put-it-behind-https)). Without one, use the server's address and the
published ports instead: `http://<server>:3000` for the frontend,
`http://<server>:8006` for the backend and `ws://<server>:8001/ws` for the
WebSocket.

| File | Setting | Example | What it is for |
|------|---------|---------|----------------|
| `autogpt_platform/.env` | `FRONTEND_BASE_URL` | `https://autogpt.example.com` | Links the backend sends people: OAuth redirects, emails, Stripe |
| `autogpt_platform/.env` | `NEXT_PUBLIC_FRONTEND_BASE_URL` | `https://autogpt.example.com` | The frontend's own address |
| `frontend/.env` | `BETTER_AUTH_URL` | `https://autogpt.example.com` | The login service's own address, used in password-reset and verification links and social sign-in callbacks |
| `frontend/.env` | `NEXT_PUBLIC_AGPT_SERVER_URL` | `https://api.autogpt.example.com/api` | The backend API, as the browser reaches it |
| `frontend/.env` | `NEXT_PUBLIC_AGPT_WS_SERVER_URL` | `wss://api.autogpt.example.com/ws` | The WebSocket server, as the browser reaches it |
| `backend/.env` | `PLATFORM_BASE_URL` | `https://api.autogpt.example.com` | Webhook and MCP URLs the backend hands to other services |
| `backend/.env` | `BACKEND_CORS_ALLOW_ORIGINS` | `["https://autogpt.example.com"]` | Origins the API and WebSocket server accept calls from (a JSON list; the default only allows `http://localhost:3000`) |

`FRONTEND_BASE_URL` and `NEXT_PUBLIC_FRONTEND_BASE_URL` are not in
`autogpt_platform/.env` yet: add both lines. Leave out the trailing `/`.

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

## 4. Put it behind HTTPS

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

`TRUSTED_PROXY_COUNT` in `backend/.env` is the number of proxies that add an
`X-Forwarded-For` entry in front of the backend. The default, `3`, is for the
hosted platform: set `1` for a single reverse proxy, or `0` with none. It
keys the per-visitor rate limits.

## 5. Decide who can sign up

Anyone who can reach the frontend can create an account. To limit that, set
one of these in `frontend/.env`:

- `AUTH_SIGNUP_ALLOWLIST`: a comma-separated list of the addresses and
  `@domains` allowed to sign up, for example `@example.com,me@example.org`.
- `AUTH_ALLOW_NEW_ACCOUNTS=false`: no new accounts at all, once yours exists.

Sign-up does not check email addresses unless you set
`AUTH_REQUIRE_EMAIL_VERIFICATION=true`. That, and password reset, need the
backend's mailer: set `POSTMARK_SERVER_API_TOKEN` and `POSTMARK_SENDER_EMAIL`
in `backend/.env`. There is no SMTP setting for login emails.

## 6. Register OAuth callback URLs

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
and `rabbitmq` containers, and set `DB_HOST`, `REDIS_HOST`, `RABBITMQ_HOST`,
`DATABASE_URL` and `DIRECT_URL` in their `environment:` sections. Those override
`backend/.env`, so pointing the stack at a managed service takes a
`docker-compose.override.yml` that sets them per service, not an `.env` change.
`DIRECT_URL` is the connection that database migrations use; give it a direct
connection, not one through a pooler such as PgBouncer. The backend expects a
Redis Cluster, not a single Redis server.

## Checklist

- [ ] `make init-env` in `autogpt_platform/`
- [ ] Your own VAPID keys in `backend/.env` and `NEXT_PUBLIC_VAPID_PUBLIC_KEY` in `frontend/.env`
- [ ] `FRONTEND_BASE_URL` and `NEXT_PUBLIC_FRONTEND_BASE_URL` added to `autogpt_platform/.env`
- [ ] `BETTER_AUTH_URL`, `NEXT_PUBLIC_AGPT_SERVER_URL` and `NEXT_PUBLIC_AGPT_WS_SERVER_URL` in `frontend/.env`
- [ ] `PLATFORM_BASE_URL` and `BACKEND_CORS_ALLOW_ORIGINS` in `backend/.env`
- [ ] A TLS reverse proxy in front of `3000`, `8001` and `8006`, those ports firewalled, and `TRUSTED_PROXY_COUNT` set
- [ ] Sign-ups limited, if the server is public
- [ ] `docker compose up -d --build`
- [ ] From another machine: sign up, log in, and check the browser console shows no `localhost` connection errors

## Related documentation

- [Getting Started](getting-started.md): the full setup, upgrades and the Makefile
- [Advanced Setup](advanced_setup.md)
- [OAuth & SSO](integrating/oauth-guide.md)
