# Server dependencies for mobile shell PR #15495

Inspected [#15495](https://github.com/Significant-Gravitas/AutoGPT/pull/15495) at `a9b6d0c1e5c3fc2b0c93d580d197e22b654e7253` against the current server baseline `6cbdbbb6b284148e53c5414bef16d0640d462768`. This is a dependency audit, not completed device/provider acceptance.

## Immediate server delivery

The shell loads the hosted web frontend in WKWebView/Android WebView. It reuses existing chat, cards, uploads and account APIs. It does not need the native renderer/discovery work planned for #15392 before basic login and chat.

Its required login server slice is already implemented in #15495:

- `frontend/src/lib/auth/mobile-auth.ts`: `/api/auth/mobile/start`, `/approve` and `/exchange`.
- `mobile-consent-page.ts`, plugin registration in `auth.ts`, and the mobile/auth-configuration tests.
- Existing `UserAuthVerification` storage for 60-second, verifier-bound, single-use grants; exchange creates a separate app session and sets HttpOnly cookies.

Those server files can be separated from the native projects and deployed first to the same HTTPS origin used by the app. The shell then needs real email/Google, consent, cancellation, cookie persistence and account-switching journeys against that deployment. This does not require a new FastAPI login API or schema migration. Google account login uses the existing browser provider flow; integration OAuth below is a different path.

The PR's auth tests use Better Auth's memory adapter. Before treating the server slice as validated for production, add PostgreSQL-backed tests with two independent auth instances racing one grant, plus expiry, source-session revocation/ban and failed exchange. The current Kysely adapter supports atomic consume through delete-and-return; a single-process memory test does not exercise that database path.

## Shared foundation

[#15384](https://github.com/Significant-Gravitas/AutoGPT/pull/15384) adds session-ID enforcement for access JWTs and pending-message retry repairs. These benefit the shell's existing web client too. The shell uses cookies, so its login handshake does not depend on native bearer transport.

Keep both auth plugins when combining changes: `bearer` and `mobile-auth`, with `nextCookies` last and JWT `sid` retained. Test the app session's JWT through the production backend, revoke that app session and confirm rejection. Signing out the source browser session after a successful exchange intentionally leaves the separate app session valid, as #15495's test specifies.

A read-only merge simulation of these heads merged the auth configuration and its tests automatically; only the secret-scanner baseline conflicted. Refresh that baseline after merging rather than dropping either plugin.

## Integration connections need a return path

The shell opens integration popups in the system browser. The ordinary provider callback at `/auth/integrations/oauth_callback` only broadcasts the authorization code via `window.opener`, `BroadcastChannel` and browser local storage. The waiting web client in `useOAuthConnect.ts` must receive that result before it calls the backend's `POST /integrations/{provider}/callback` exchange.

Those browser channels cannot reach the separate app WebView. This is a concrete gap for ordinary provider connections; a successful Google account login does not validate connecting Google or another integration.

A focused server follow-up can persist an OAuth attempt and its completion, bind it to the initiating user/session/provider and existing credential, redeem the callback once, and let the authenticated app poll status on return. Any native return link should only wake the app; server state remains authoritative. Test cancel, timeout, account switching, duplicate callbacks, denied grants and reconnect preserving credential identity. Pair this with a client resume/status update. This is a proposed implementation boundary, not an endpoint already present.

## Later capabilities

- **Native push:** current `PushSubscription` stores Web Push endpoint/`p256dh`/`auth`, and current routes expose VAPID subscriptions. Native APNs/FCM registration, rotation/revocation and delivery are separate backend work, paired with client permission and routing.
- **Store purchases:** the shell currently exposes the existing hosted billing UI. If native purchases are enabled, add provider verification, notification processing and shared entitlement reconciliation first. Store/payment policy selection is a separate product decision.
- **Offline delivery:** do not enable automatic native outbox replay on the assumption that the pending-message fixes provide durable cancellation. The restrictions in the [backend handoff](backend-handoff.md) still apply.
- **Downloads and lifecycle:** browser downloads, WebView cookie transfer, microphone/file permissions and stream resume require device tests. A new backend endpoint is not established as necessary merely because those checks remain open.

Recommended order: isolate and validate the existing server login handoff; combine and validate session enforcement; implement a mobile integration return/completion path; add push and purchase backends when those features are enabled. Full native card contracts remain specific to the separate native-rendered client plan.
