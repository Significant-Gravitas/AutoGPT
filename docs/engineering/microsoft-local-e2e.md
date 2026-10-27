# Microsoft connection and sign-in

One Entra app per environment does everything Microsoft: the Teams bot, the
Microsoft 365 Copilot chat route, mail/calendar/files/Teams access for Otto
and workflows, Work IQ, and "Continue with Microsoft" on the login page.

## What a person sees

**Continue with Microsoft** signs in with the account picker and asks only for
name and email. A new account's onboarding then opens with an access step that
offers the data connection; skipping it is fine. Existing accounts go straight
where they were heading and can connect from Settings.

**Connect Microsoft** (Settings → Integrations, the Copilot card, or the access step) creates
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
  panel that Settings, the Microsoft 365 Copilot card and the access step
  share. A personal and a work account can share an address and stay
  separate connections.
- `frontend/src/app/(no-navbar)/onboarding/steps/AccessStep/`: the access
  step. `accessProviders.tsx` maps a sign-in provider to the connection it
  offers; Google or GitHub join by adding an entry there.
- `frontend/src/lib/auth/microsoft-login.ts`: the login provider. Accounts
  are keyed on tenant plus Microsoft's object ID, never the email address.
  An address only counts when Microsoft vouches for it: a personal account's
  own Outlook.com, Hotmail, Live or MSN address, or a work address with
  `xms_edov`. A guest's address is never trusted: a guest signs in through
  an organization that isn't its own, which vouches for addresses it never
  verified. Any other address joins no existing account (Better Auth
  answers `account_not_linked`) and creates none
  (`microsoft-sign-up-policy.ts` answers `unconfirmed_email`), whether or not
  email verification is on. Only the redirect flow signs in: Better Auth's
  ID-token sign-in is switched off for Microsoft. Emailing a confirmation link instead would not be
  safe: the mailbox's owner could confirm it, and the other person's
  Microsoft login would still sign in to the account.

## Moving the access step

The wizard's order is data (`DEFAULT_STEP_ORDER` in
`onboarding/store.ts`), so an experiment can move a step without a code
change. The `onboarding-step-order` flag's payload is a JSON list of step
names. The steps it names lead, in its order; every other step follows in the
default order, and `preparing` is always last. It only moves steps: which ones
show still depends on the account (the access step needs a provider sign-in,
the paywall needs payments on). For example:

- `[]` (or no flag): access, team, autopilot, role, painPoints, hire, ...
- `["team", "autopilot", "role", "access"]`: the access step after the role.

The order is fixed when the wizard loads, so changing the flag never
reorders it under someone. Progress is saved by step name, so a later visit
resumes at the right step even if the order changed in between. New steps,
such as recommendations drawn from connected services where each service's
terms allow it, slot into the same list.

## Entra app settings (per environment)

On the environment's app (dev: `autogpt-dev`, prod: `AutoGPT`):

1. **Manifest:** `api.requestedAccessTokenVersion` = `2`, then
   `signInAudience` = `AzureADandPersonalMicrosoftAccount`. The order matters;
   the audience change fails without token version 2. The Teams bot is
   unaffected: it authenticates against the home tenant with its own secret.
2. **Web redirect URIs:** `{frontend}/auth/integrations/oauth_callback` (data
   connection and organization approval) and
   `{frontend}/api/auth/callback/microsoft` (Continue with Microsoft).
3. **Token configuration:** add the optional ID-token claim `xms_edov`.
   Without it, work and school accounts cannot use Continue with Microsoft.
4. **Certificates & secrets:** the bot's existing secret also serves sign-in.
   Rotate by adding a new secret, deploying it, then removing the old one.

Then set `MICROSOFT_BROWSER_SIGN_IN=true`. A secret alone doesn't turn browser
sign-in on, because the bot's app has one long before it has the redirect.
Until the flag is on, Microsoft connects with sign-in codes through the same
app. Continue with Microsoft has its own switch, below.

Dev's app is configured (2026-10-26) but still needs the flag; prod needs
steps 1–3 and then the flags.

## Environment variables

| Where | Setting | Notes |
|---|---|---|
| backend | `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET`, `MICROSOFT_TENANT_ID` | Shared with the Teams bot. Without a secret, connections use sign-in codes. |
| backend | `MICROSOFT_BROWSER_SIGN_IN=true` | Browser sign-in instead of sign-in codes, once the app has the redirect above. |
| frontend | `AUTH_MICROSOFT_CLIENT_ID`, `AUTH_MICROSOFT_CLIENT_SECRET` | Same app as the backend. |
| frontend | `NEXT_PUBLIC_AUTH_MICROSOFT_ENABLED=true` | Shows the login button. |

## Installs without an app secret

`.env.default` ships no Microsoft settings. The backend then uses AutoGPT's
public client (`ce01a4d6-…`) with sign-in codes, so a fresh local install can
still connect Microsoft and use Copilot. Browser sign-in and Continue with
Microsoft need an app with a secret, the redirects above and their switches.

## Testing locally

Run the normal AutoGPT stack behind one private HTTPS origin (for example a
Tailscale listener), set the frontend and backend public URLs to it, and
register its two callbacks on the dev app. Then:

1. Log in, open Settings → Integrations → Connect Service → Microsoft, and
   sign in. Check each service's status, then try **Allow everything listed**
   on a work account.
2. Open the Microsoft 365 Copilot card; a work connection with Copilot's
   permissions should already count as connected.
3. Log out and use **Continue with Microsoft** with a personal and a work
   account. A new account's onboarding should open on "Also connect Microsoft
   365?".

Work IQ also needs Microsoft's own setup in the tenant (the
`fdcc1f02-…` service, admin approval and Copilot Credits billing). Until then
its status is "Your organization still needs to set this up".
