# OAuth Flow — Shim Authentication

> **Status**: Spec / Not Implemented

The shim authenticates using AutoGPT's **existing OAuth 2.0 provider infrastructure** —
the same system used by third-party app integrations. No new auth infrastructure needed.

## Why Not Device OAuth?

AutoGPT already runs a full OAuth 2.0 Authorization Server:
- `/auth/authorize` — authorization endpoint
- `/auth/token` — token endpoint  
- `introspect_token()` — token validation
- Authorization Code + PKCE flow already implemented

Device OAuth would be redundant. The shim registers as an OAuth app and uses
Authorization Code + PKCE with a localhost redirect URI.

---

## Registration (One-Time)

The shim is a first-party OAuth application registered in the AutoGPT
platform. **v0 uses a confidential client + PKCE flow** (not pure-public
PKCE). Rationale: the platform's existing OAuth schema requires a
`clientSecret` on every grant, and true public-client support is a
follow-up schema change. PKCE still provides session security; the
embedded secret is published with the shim source.

```
Client ID:     autogpt-local-executor (well-known)
Client Secret: published with the shim distribution (PKCE provides
               session security; treat the secret as public)
Redirect URI:  http://localhost:{port}/callback   where {port} ∈ 41899..41910
Scopes:        local_executor:connect local_executor:shell local_executor:files
               (optional) local_executor:computer_use local_executor:hardware
Grant types:   authorization_code, refresh_token
```

Ports 41899–41910 (12 ports) are reserved for the shim's local callback
server. The shim binds the first free port in that range, so a busy 41899
(held by another app, a stale shim process, or a developer's local dev
server) doesn't break first-time auth. The platform's OAuth client
registration MUST accept all ports in this range as valid `redirect_uri`
values for `client_id=autogpt-local-executor` — registering only 41899
would defeat the fallback. Listener is always bound to `127.0.0.1`; never
to `0.0.0.0`. See [CROSS_PLATFORM.md → OAuth callback port](CROSS_PLATFORM.md#oauth-callback-port).

### Operator setup (per-environment)

Each platform deployment registers the app once via the existing
`oauth-tool` CLI inside the backend container:

```bash
poetry run oauth-tool generate-app \
    --name "AutoGPT Local Executor" \
    --description "Local PC shim for the AutoGPT hosted platform" \
    --redirect-uris \
      "http://localhost:41899/callback,http://localhost:41900/callback,\
http://localhost:41901/callback,http://localhost:41902/callback,\
http://localhost:41903/callback,http://localhost:41904/callback,\
http://localhost:41905/callback,http://localhost:41906/callback,\
http://localhost:41907/callback,http://localhost:41908/callback,\
http://localhost:41909/callback,http://localhost:41910/callback" \
    --scopes "EXECUTE_GRAPH"
```

The tool prints a generated `client_id` (format `agpt_client_<token>`)
and `client_secret`. Until `oauth-tool` learns a `--client-id` flag,
the "well-known `autogpt-local-executor`" id is aspirational — each
platform deployment will have its own random client_id. The shim's
distribution config (`AUTOGPT_LOCAL_EXECUTOR_CLIENT_ID` +
`_CLIENT_SECRET`) is baked at build time from the operator's chosen
deployment. End-users installing the official shim get the canonical
agpt.co client_id; self-hosters bake in their own.

### Future: true public-client support

To match the OAuth 2.1 best-practice for desktop apps (PKCE, no
client_secret), the platform schema needs an `isPublic` boolean on
`OAuthApplication` and `validate_client_credentials` must short-circuit
the secret check for public clients on PKCE grants. Tracked as a
follow-up — not blocking shim MVP.

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
 |                       |      scope=local_executor:connect ...
 |                       |                                |
 |     [Browser opens]   |                                |
 |<====================================================-->|
 |     [User logs in and approves scopes]                 |
 |                       |                                |
 |                       |<-- GET /callback?code=AUTH_CODE
 |                       |                                |
 |                       | 4. POST /auth/token            |
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
GET /ws/local-executor/{session_id}
Authorization: Bearer {access_token}
```

Platform validates via `introspect_token(access_token)`:
- Checks token not expired
- Checks token belongs to the session owner
- Checks `local_executor:connect` scope present
- Returns user_id for the session

---

## Token Refresh

The shim manages token refresh proactively:
- Refresh 5 minutes before expiry using stored `refresh_token`
- On 401 during WebSocket upgrade, refresh and retry once
- On refresh failure (expired refresh token), prompt user to re-auth via CLI: `autogpt-shim auth`

Tokens stored in OS keychain:
- macOS: Keychain Services via `keyring` library
- Linux: Secret Service (GNOME Keyring / KWallet) via `keyring`
- Windows: Windows Credential Manager via `keyring`

---

## Per-Capability Scopes

| Capability | Required Scope |
|-----------|----------------|
| Shell execution | `local_executor:shell` |
| File read/write | `local_executor:files` |
| Computer use | `local_executor:computer_use` |
| Hardware access | `local_executor:hardware` |
| Local LLM | `local_executor:local_llm` |
| Background tasks | `local_executor:background` |

The platform only grants scopes the user explicitly approved during OAuth.
The shim only advertises capabilities it has scopes for.
