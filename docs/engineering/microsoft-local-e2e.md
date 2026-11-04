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
- Work IQ is never asked for together with anything else, neither in "Allow
  everything listed" nor in the organization approval: until Microsoft has set
  Work IQ up for the organization, asking for it fails everything asked with
  it (`AADSTS650052`), and nothing shows that before someone asks. Its
  permission is admin-consent only (Microsoft marks `WorkIQAgent.Ask`
  AdminConsentRequired), so it shows "needs your admin" until an admin
  approves it, from its own **Allow** or for the whole organization. After
  that every existing connection gets Work IQ from its own refresh token, with
  no new sign-in.
- If Microsoft stops accepting a connection (an expired or revoked grant), the
  panel asks the person to sign in again.

Copilot chats run on a work connection that has Copilot's permissions; there
is no separate Copilot sign-in. Copilot connections made before this keep
working until they are disconnected.

## What each permission is for

The panel, the Copilot card and the admin setup page (`/microsoft-setup`) say
this wherever someone grants a permission.

| Service | Permissions | What it lets AutoGPT do |
|---|---|---|
| Mail | `Mail.ReadWrite`, `Mail.Send` | Otto reads, drafts and sends the person's email. |
| Calendar | `Calendars.ReadWrite` | Otto sees, adds and changes their calendar events. |
| Files | `Files.ReadWrite` | Otto opens, creates and edits their OneDrive files. |
| Teams chats | `Chat.ReadWrite` | Otto reads and sends messages in their Teams chats. |
| Microsoft 365 Copilot | Copilot's read permissions (`Sites.Read.All`, `Mail.Read`, `Chat.Read`, …) | Microsoft 365 Copilot answers chats routed to it, from what the account can open. |
| Work IQ | `WorkIQAgent.Ask` | Otto asks Work IQ, which can read and change the person's Microsoft 365 content. |

Every sign-in also asks for `User.Read` and `openid profile email
offline_access`, to identify the account and keep it connected. Today the
Copilot chat route is the only thing that runs on the connection; Otto's mail,
calendar, files, Teams and Work IQ tools follow it.

## How it works

- `backend/integrations/oauth/microsoft.py`: browser sign-in (authorization
  code with PKCE) and the sign-in-code fallback. Both produce the same
  credential, recording which app issued it.
- `microsoft_tokens.py`: token requests. Every request goes out as the app
  that issued the grant; only the deployment's own app sends its secret.
  Work IQ tokens are minted from the same refresh token: a Microsoft refresh
  token isn't tied to one resource, and gets tokens for any resource the app
  has consent for, including consent an admin gives after the sign-in.
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
4. **API permissions:** add the connection's delegated Microsoft Graph
   permissions: `User.Read`, `openid`, `profile`, `email`, `offline_access`,
   `Mail.ReadWrite`, `Mail.Send`, `Calendars.ReadWrite`, `Files.ReadWrite`,
   `Chat.ReadWrite`, and Copilot's `Sites.Read.All`, `Mail.Read`,
   `People.Read.All`, `OnlineMeetingTranscript.Read.All`, `Chat.Read`,
   `ChannelMessage.Read.All` and `ExternalItem.Read.All`. Sign-ins ask for
   their permissions as they go, so this changes nothing for them. It is what
   an admin approves when they grant consent outside AutoGPT: the Entra admin
   center and Microsoft's `/adminconsent?client_id=…` link, which
   `/microsoft-setup` points to in sign-in-code mode, approve only the
   permissions listed here. Leave Work IQ out: listed, it would fail that
   approval wherever Microsoft hasn't set Work IQ up (`AADSTS650052`).

Then set `MICROSOFT_BROWSER_SIGN_IN=true`. A secret alone doesn't turn browser
sign-in on, because the bot's app has one long before it has the redirect.
Until the flag is on, Microsoft connects with sign-in codes through the same
app, at `/common` (the Copilot-only connection used `/organizations`). So step
1 is needed before an environment gets this release, not only before the
flag: without it, personal accounts can't sign in.

Dev's app has steps 1–3 (2026-10-26); step 4 and the flag remain. Prod needs
step 1 before this release reaches it, and step 4 so an admin's approval
covers every service; then step 2 and the flag.

## Environment variables

| Where | Setting | Notes |
|---|---|---|
| backend | `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET`, `MICROSOFT_TENANT_ID` | Shared with the Teams bot. Without a secret, connections use sign-in codes. |
| backend | `MICROSOFT_BROWSER_SIGN_IN=true` | Browser sign-in instead of sign-in codes, once the app has the redirect above. |

## Installs without an app secret

`.env.default` ships no Microsoft settings. The backend then uses AutoGPT's
public client (`ce01a4d6-…`) with sign-in codes, so a fresh local install can
still connect Microsoft and use Copilot. Browser sign-in needs an app with a
secret, the redirect above and `MICROSOFT_BROWSER_SIGN_IN=true`.

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
