# Microsoft connection

One Entra app per environment does everything Microsoft: the Teams bot, the
Microsoft 365 Copilot chat route, mail/calendar/files/Teams access for Otto
and workflows, and Work IQ.

## What a person sees

**Connect Microsoft** (Settings → Integrations, or the Copilot card) creates
one `microsoft` connection per Microsoft account:

- Personal accounts get mail, calendar and files.
- Work or school accounts can add Teams chats, Microsoft 365 Copilot and
  Work IQ. Each service reports its own status: ready, needs your permission
  (one **Allow** click, or **Allow everything listed**), needs your admin,
  needs organization setup (Work IQ), or work accounts only.
- An admin can choose **Approve for my whole organization**, after which
  everyone in that tenant gets every service in one click.

Copilot chats run on a work connection that has Copilot's permissions; there
is no separate Copilot sign-in. Copilot connections made before this keep
working until they are disconnected.

## How it works

- `backend/integrations/oauth/microsoft.py`: browser sign-in (authorization
  code with PKCE) and the sign-in-code fallback. Both produce the same
  credential, recording which app issued it.
- `microsoft_tokens.py`: token requests. Every request goes out as the app
  that issued the grant; only the deployment's own app sends its secret.
  Work IQ tokens are minted from the same refresh token, because Microsoft
  refresh tokens cover every permission the person consented to.
- `microsoft_services.py`: services and their permissions. A sign-in may ask
  consent for Graph and Work IQ together; the stored token is always Graph.
- `microsoft_access.py`: the per-service status and the organization
  approval link (`POST /api/integrations/microsoft/check`).
- `frontend/src/components/contextual/MicrosoftConnection/`: the connection
  panel that Settings and the Microsoft 365 Copilot card share. A personal
  and a work account can share an address and stay separate connections.

## Entra app settings (per environment)

On the environment's app (dev: `autogpt-dev`, prod: `AutoGPT`):

1. **Manifest:** `api.requestedAccessTokenVersion` = `2`, then
   `signInAudience` = `AzureADandPersonalMicrosoftAccount`. The order matters;
   the audience change fails without token version 2. The Teams bot is
   unaffected: it authenticates against the home tenant with its own secret.
2. **Web redirect URI:** `{frontend}/auth/integrations/oauth_callback`, for
   the connection and the organization approval.
3. **Certificates & secrets:** the bot's existing secret also serves browser
   sign-in. Rotate by adding a new secret, deploying it, then removing the
   old one.

Dev is configured (2026-10-26); prod still needs steps 1 and 2.

## Environment variables

| Where | Setting | Notes |
|---|---|---|
| backend | `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET`, `MICROSOFT_TENANT_ID` | Shared with the Teams bot. Without a secret, connections use sign-in codes. |

## Installs without an app secret

`.env.default` ships no Microsoft settings. The backend then uses AutoGPT's
public client (`ce01a4d6-…`) with sign-in codes, so a fresh local install can
still connect Microsoft and use Copilot. Browser sign-in needs an app with a
secret and the redirect above.

## Testing locally

Run the normal AutoGPT stack behind one private HTTPS origin (for example a
Tailscale listener), set the frontend and backend public URLs to it, and
register its callback on the dev app. Then:

1. Log in, open Settings → Integrations → Connect Service → Microsoft, and
   sign in. Check each service's status, then try **Allow everything listed**
   on a work account.
2. Open the Microsoft 365 Copilot card; a work connection with Copilot's
   permissions should already count as connected.

Work IQ also needs Microsoft's own setup in the tenant (the
`fdcc1f02-…` service, admin approval and Copilot Credits billing). Until then
its status is "Your organization still needs to set this up".
