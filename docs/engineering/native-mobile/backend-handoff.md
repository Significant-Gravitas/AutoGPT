# Native mobile backend foundation and client handoff

The backend foundation in [#15384](https://github.com/Significant-Gravitas/AutoGPT/pull/15384) can be reviewed and landed before the native client in [#15392](https://github.com/Significant-Gravitas/AutoGPT/pull/15392). This boundary covers email session authentication and pending-message identity. The remaining server work and native acceptance gates below still apply to the full experiment.

The same session and retry contracts can be used by installed mobile and desktop clients. Client interfaces, browser login returns and distribution remain separate work.

The pending-message changes ship with session authentication because reconnecting clients retry accepted sends. Preserving their IDs requires recovery to retain the same IDs through cancellation, database failure and buffer refill. The engine changes preserve existing approval results and transient-error behavior when those recovery boundaries are introduced.

## Foundation landing boundary

- Better Auth accepts signed session bearers and issues access JWTs containing `sid`.
- Backend REST requests and new WebSocket connections validate that session's owner, expiry, revocation and account ban. The production REST validator survives test dependency-override cleanup.
- Pending messages retain their scoped client ID through queue admission and database persistence. Full queues reject new messages without evicting accepted ones. Persistence rollback restores the drained batch ahead of newer arrivals, even after the buffer refills. Receipt recovery and persisted-row reconciliation prevent the covered duplicate retry cases.
- The dedicated Ubuntu lane exercises production auth over HTTP and real Redis/PostgreSQL retry boundaries. It requires no mobile build, live model account, push key or payment account.

Server-side Better Auth code lives under `autogpt_platform/frontend/`; it belongs to this foundation even when the native UI is handed off separately. No application-data migration or new production secret is required.

The foundation targets `dev` directly. Landing requires current-base checks and review of authentication and existing web-client compatibility. Native simulator timing, complete native cards and provider certification belong to the client/product gates, not this foundation's merge checklist. Merging, deploying and enabling a native feature remain separate actions.

## Contract the client can use

| Boundary                 | Client behavior                                                                                                                                                                                                                                            |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Email sign-in            | Use the selected server's production `/api/auth/sign-in/email` with `X-AutoGPT-Native: 1`, without cookies, Origin or browser Fetch Metadata headers. Store the signed `set-auth-token` response header in Keychain, partitioned by server and account. The unsigned token in the JSON body is not a bearer credential.                          |
| Access token             | Send the signed session credential only to that server's auth service. Exchange it at `GET /api/auth/token`; send the returned JWT to backend APIs. Renew through the auth service when needed.                                                            |
| Sign-out and revocation  | Sign out through the auth service. A revoked, expired, banned or mismatched session is rejected on subsequent backend authentication. Clear local credentials and account-scoped state. Existing streams are not forcibly terminated.                      |
| Auth service unavailable | HTTP 503 with `Retry-After: 5` means retry authentication later, without treating the account as signed out. New WebSocket authentication closes with 1013 for this condition.                                                                             |
| Pending send identity    | Generate one nonempty `message_id` of at most 64 characters per user send. Reuse it for retries of that send; a distinct send gets a new ID even when its text is identical. The server scopes it to the user and conversation.                            |
| Pending send admission   | `POST /api/chat/sessions/{session_id}/messages/pending` returns the queue response. A full queue is HTTP 429; no active turn is HTTP 409. HTTP 503 with `Retry-After` means the service could not establish admission. Preserve the send ID when retrying. |
| Accepted retry           | A successful repeated admission is an acknowledgement, not proof that the message is still in the queue or that a turn is currently running. Refresh persisted history and stream state.                                                                   |
| Ownership                | Continue using the selected server, authenticated account and conversation together. Server authorization remains authoritative; a client ID does not grant access to a conversation.                                                                      |

JWKS connection failures follow the retryable authentication-outage contract above. WebSocket codes 4001, 4002 and 4003 reject missing or invalid credentials; 4004 rejects a banned account. Stop automatic reconnects for these codes. Ban-lookup failures retain dev's retryable 1011 close code.

An account its owner closed (self-serve deletion) gets HTTP 401 with detail `This account is closed.` from signed-in routes other than the account-deletion status and restore routes, and WebSocket 4003 with that reason. A failed closed-account lookup closes with retryable 1011, as a failed ban lookup does.

**Keep automatic outbox replay disabled.** Redis receipts expire, cancellation is not durably fenced after receipt expiry or Redis loss, and a Redis outage can still interrupt rollback recovery. Do not promise durable offline delivery or replay an old send automatically on relaunch/account switching. A later backend slice must prove cancellation remains effective when a delayed request crosses cancellation and a subsequent turn, including database failures and Redis loss.

## Remaining backend slices

| Slice                         | Required contract and evidence                                                                                                                                  |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Durable send/cancellation     | Durable admission/cancellation authority, recovery after Redis loss, delayed-request races and persistence failures; required before automatic outbox replay.   |
| Discovery and card actions    | Versioned capabilities, complete card/event fixtures, stable question/review identity, stale/two-device rejection and exactly one accepted continuation.        |
| Google and connection handoff | Registered browser return destinations, account/server binding, cancellation and reconnect behavior; credential edits preserve identity, grants and references. |
| Push                          | Authoritative session/device registration, revocation and minimal notification content; real provider delivery remains a separate test.                         |
| Subscription entitlements     | Verified provider events, account binding, retry/order handling and reconciliation of Stripe/App Store grants before native purchases are enabled.              |

These are follow-up backend changes, not claims about functionality already present in #15384. Product planning is tracked in [SECRT-2881](https://linear.app/autogpt/issue/SECRT-2881).

## Validation and rollout

Use the isolated database and Redis instructions in [server validation](server-validation.md). The foundation regression set covers real signed email sessions, token renewal, logout, expiry, bans, ownership mismatch, lookup failure/timeout, unsigned bearer rejection, concurrent retry, overflow, receipt/list expiry and real PostgreSQL primary-key collision recovery.

Unset `AUTH_LEGACY_TOKEN_DEADLINE` preserves signature/expiry validation for existing JWTs without `sid`, so independently deployed auth and API services remain compatible. Tokens containing `sid` always receive session checks, including during rollout. Deploy the signer, then choose one fixed Unix cutoff shared by all REST and WebSocket servers at least one access-token lifetime (currently one hour) after the signer rollout completes. Set `AUTH_LEGACY_TOKEN_DEADLINE` to that cutoff; `0` enforces session IDs immediately. Never move the cutoff on worker restarts.

**Breaking change when the cutoff is enabled:** JWTs without `sid` receive HTTP 401 with `X-Error-Code: session_token_renewal_required`, or WebSocket 1013. Renew through the auth service; the web client obtains a fresh JWT on reconnect. Invalid or revoked sessions still receive terminal authentication rejection. Service tokens retain their separate audience and verification path.

Browser requests retain HttpOnly session cookies and never receive the native `set-auth-token` response header. Native credential export requires the explicit header above and refuses cookie-bearing or browser-originated requests. Native clients should keep sending that header when renewing their session and keep bearer credentials in OS secure storage.

Session authority uses an uncached indexed PostgreSQL lookup per authenticated request and new WebSocket connection. The same bearer shares validation within one request only. An unavailable auth database returns 503/1013 instead of allowing a request through with signature validation alone; provision and monitor database capacity accordingly.

Optional-auth store and marketplace routes preserve this distinction: a request with no token remains anonymous, and an invalid or revoked token falls back to anonymous access. A supplied token that cannot be verified because JWKS or session authority is unavailable returns 503 with retry guidance. This is intentional: an authentication outage must not silently turn an authenticated request into an anonymous one. Clients that explicitly want anonymous browsing can omit the bearer token.

Rejected WebSocket authentication now completes the protocol upgrade before sending its close code. Proxy access logs therefore count those attempts as upgraded connections; authenticated-connection metrics still start only after authentication. Existing perimeter rate limits should include the WebSocket route.

The signer change affects web-issued JWTs too, so include existing web sign-in, token renewal and request flows in rollout checks. Rolling back the validator removes the new revocation guarantee for already issued tokens; treat that explicitly as a rollback limitation.

The native app remains independently gated on complete actions, rendered iOS/design-system review, lifecycle/accessibility coverage, and measured cold and warm standard-runner journeys below ten minutes. A passing server lane or native compile is not evidence that those gates passed.
