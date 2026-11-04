# OAuth Flow — Shim Authentication

> **Status**: Implemented

The shim authenticates using AutoGPT's **existing OAuth 2.0 provider infrastructure** —
the same system used by third-party app integrations. No new auth infrastructure needed.

## Why Not Device OAuth?

AutoGPT already runs a full OAuth 2.0 Authorization Server:
- `/auth/authorize` — authorization endpoint
- `/api/oauth/token` — token endpoint
- `introspect_token()` — token validation
- Authorization Code + PKCE flow already implemented

Device OAuth would be redundant. The shim registers as an OAuth app and uses
Authorization Code + PKCE with a localhost redirect URI.

---

## Registration (One-Time)

The shim is a first-party public OAuth application registered in the AutoGPT
platform. It uses Authorization Code + PKCE and has no client secret.

```
Client ID:     autogpt-local-executor (well-known)
Client type:   public (PKCE; no client secret)
Redirect URI:  http://localhost:{port}/callback   where {port} ∈ 41899..41910
Scopes:        CONNECT_LOCAL_EXECUTOR
Grant types:   authorization_code, refresh_token
```

Ports 41899–41910 (12 ports) are reserved for the shim's local callback
server. The shim binds the first free port in that range, from
`oauth_redirect_port` on and never past 41910 (a configured port outside the
range is the only one it tries, since no other is known to be registered), so a busy 41899
(held by another app, a stale shim process, or a developer's local dev
server) doesn't break first-time auth. The platform's OAuth client
registration MUST accept all ports in this range as valid `redirect_uri`
values for `client_id=autogpt-local-executor` — registering only 41899
would defeat the fallback. Listener is always bound to `127.0.0.1`; never
to `0.0.0.0`. See [CROSS_PLATFORM.md → OAuth callback port](CROSS_PLATFORM.md#oauth-callback-port).
The server takes one connection at a time, so each read waits at most 10
seconds and a whole request at most 20 before the connection is dropped: a
browser's preconnect that never sends, or a client that sends a byte now and
then, can't hold off the redirect. A request to `/callback` whose `state` isn't
this sign-in's is refused with 400 and doesn't end the wait, so neither a tab
left from an earlier attempt nor a page that requests the URL can end the
sign-in. On Windows the server takes its port exclusively
(`SO_EXCLUSIVEADDRUSE`), so a port another program listens on counts as busy
and the next one is tried.

### Operator setup (per-environment)

Each platform deployment registers the app once via the existing
`oauth-tool` CLI inside the backend container:

```bash
poetry run oauth-tool generate-app \
    --name "AutoGPT Local Executor" \
    --client-id "autogpt-local-executor" \
    --public \
    --description "Local PC shim for the AutoGPT hosted platform" \
    --redirect-uris \
      "http://localhost:41899/callback,http://localhost:41900/callback,\
http://localhost:41901/callback,http://localhost:41902/callback,\
http://localhost:41903/callback,http://localhost:41904/callback,\
http://localhost:41905/callback,http://localhost:41906/callback,\
http://localhost:41907/callback,http://localhost:41908/callback,\
http://localhost:41909/callback,http://localhost:41910/callback" \
    --scopes "CONNECT_LOCAL_EXECUTOR"
```

The command prints SQL containing the well-known public client ID and an
`isPublic=true` marker. Replace `YOUR_USER_ID_HERE`, execute it once per
environment, and keep every callback URI in the registration. Self-hosters
that choose another ID configure it with `AUTOGPT_SHIM_OAUTH_CLIENT_ID`.

### True public-client support (implemented)

`OAuthApplication.isPublic` (added in migration
`20260604120000_add_oauth_application_is_public`) and the
`validate_client_credentials(..., code_verifier=...)` short-circuit
together implement the OAuth 2.1 best-practice for desktop / mobile /
SPA clients (PKCE, no client_secret).

- Confidential client (default, `isPublic=false`): `client_secret` is
  required as before. No behavior change.
- Public client (`isPublic=true`): `client_secret` is ignored;
  `code_verifier` is required and is the only credential. The PKCE
  challenge/verifier check still happens in `consume_authorization_code`
  (which compares the verifier against the stored code_challenge).

The `oauth-tool generate-app --public` command sets this field for new shim
registrations. Existing registrations can be migrated with `UPDATE
"OAuthApplication" SET "isPublic" = true WHERE "clientId" = '<id>'`.

Introspect (`/api/oauth/introspect`) remains confidential-auth-only. Revoke
(`/api/oauth/revoke`) accepts this registered public client without a secret,
binds each token to the client before revoking it, and returns the RFC 7009
non-disclosing success response. `autogpt-shim revoke` revokes the access and
refresh tokens for the configured deployment before clearing that deployment's
keychain record. Other deployments' records are not removed. The platform also pushes a
`SESSION_REVOKED` frame to a connected shim (see `PROTOCOL.md`).

---

## First-Time Auth Flow

```
User                    Shim                         AutoGPT Platform
 |                       |                                |
 | autogpt-shim auth     |                                |
 |---------------------->|                                |
 |                       | 1. Generate code_verifier      |
 |                       |    code_challenge = S256(cv)   |
 |                       |                                |
 |                       | 2. Spin up localhost:41899      |
 |                       |                                |
 |                       | 3. Open browser →              |
 |                       |    /auth/authorize?             |
 |                       |      client_id=autogpt-local-executor
 |                       |      redirect_uri=http://localhost:41899/callback
 |                       |      code_challenge=...        |
 |                       |      scope=CONNECT_LOCAL_EXECUTOR
 |                       |                                |
 |     [Browser opens]   |                                |
 |<====================================================-->|
 |     [User logs in and approves scopes]                 |
 |                       |                                |
 |                       |<-- GET /callback?code=AUTH_CODE
 |                       |                                |
 |                       | 4. POST /api/oauth/token        |
 |                       |      grant_type=authorization_code
 |                       |      code=AUTH_CODE            |
 |                       |      code_verifier=...         |
 |                       |                                |
 |                       |<-- {access_token, refresh_token, expires_in}
 |                       |                                |
 |                       | 5. Store tokens in OS keychain |
 |                       |    (keyring library)           |
 |    Auth complete       |                                |
 |<----------------------|                                |
```

---

## WebSocket Connection Auth

On every WebSocket connect, the shim includes the access token:

```
GET /ws/local-executor                 # persistent machine control
GET /ws/local-executor/{session_id}    # activated per-chat data channel
Authorization: Bearer {access_token}
```

Platform validates via `introspect_token(access_token)`:
- Checks token not expired
- On data channels, checks token belongs to the session owner
- On the control channel, registers the online machine under the token's user
- Checks `CONNECT_LOCAL_EXECUTOR` scope present (HTTP 403 without it)
- Returns user_id for the session

---

## Token Refresh

The shim refreshes using the stored `refresh_token` exactly once when the
initial WebSocket upgrade returns 401. An initial 403 is treated as a terminal
session-authorization denial and never rotates credentials. If the post-refresh
retry returns 401 or 403, startup terminates instead of entering the reconnect
loop. The refresh is forced even while the rejected access token remains in the
keychain.
- On refresh failure (expired refresh token), prompt user to re-auth via CLI: `autogpt-shim auth`

Tokens stored in OS keychain:
- macOS: Keychain Services via `keyring` library
- Linux: Secret Service (GNOME Keyring / KWallet) via `keyring`
- Windows: Windows Credential Manager via `keyring`

### Deployment-bound credentials

Access and refresh tokens are stored together in a versioned keychain record.
Its key is derived from the configured backend base, OAuth client ID, browser
authorization URL, token/revocation endpoints, and WebSocket base. The same
endpoint metadata is checked when reading the record. Backend path prefixes
such as `/backend` remain part of the identity. Host casing, default ports,
equivalent HTTP/WebSocket base schemes, and trailing base slashes do not create
duplicate identities; different hosts, non-default ports, clients, and endpoint
paths do. The per-chat session ID is not part of this shared machine credential.

Split frontend/backend deployments are supported: `--platform-oauth-url` selects
the browser authorization base while `--platform-url` selects the backend.
Explicit OAuth or WebSocket overrides are bound when authenticating. Changing
one does not authorize sending previously saved tokens to the new destination:
authenticate again with the intended configuration. OAuth requests do not follow
HTTP redirects; WebSocket redirects, including same-origin path changes, are
rejected. A destination change while the daemon or OAuth flow is running is a
terminal authentication error, not a refresh attempt against the new endpoint.

Use the same persistent configuration for every command. A `--config` that
names no file is an error rather than a quiet fallback to the defaults:

```bash
autogpt-shim --config /path/to/config.toml auth
autogpt-shim --config /path/to/config.toml start
autogpt-shim --config /path/to/config.toml revoke
```

Older releases used global `access_token` and `refresh_token` keychain entries
without deployment metadata. Their origin cannot be inferred safely. The shim
never forwards, migrates, or deletes those legacy entries automatically. Startup,
refresh, and revocation fail closed if only legacy credentials are available,
with instructions to authenticate explicitly. Successful `auth` creates a new
scoped record and leaves legacy entries untouched. Revoke the legacy authorization
at its original platform deployment; after doing so, remove the old entries using
the OS credential manager if desired. A corrupt or mismatched scoped record also
requires explicit authentication rather than falling back to legacy credentials.

---

## Scope

The public shim client requests `CONNECT_LOCAL_EXECUTOR`, the AutoGPT
`APIKeyPermission` for exactly this connection. It grants nothing else: no
External API tools, and API keys cannot carry it. The WebSocket handshake
requires that scope in addition to an active access token and session
ownership. A token issued before the scope existed is refused with HTTP 403, and
the shim asks for `autogpt-shim auth` to be run again. Individual executor capabilities
remain opt-in through shim configuration, HELLO capability advertisement,
platform feature flags, and the per-session computer-use consent gate.
