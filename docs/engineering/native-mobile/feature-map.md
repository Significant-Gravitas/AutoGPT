# Native mobile feature map

Status: source map reviewed; server implementation started. Baseline: `81d43f2982fe406acb7648f19c131e491c365d64` on `dev`, inspected on 2026-10-11. This experiment replaces the retired web-host approach in PR #14429. iOS is first; Android follows. The AI company means Otto and hired experts, not organization membership administration.

Fable supplied and reviewed the backend/tool inventory; Opus supplied and reviewed the renderer/client inventory and architecture. The parent independently verified the enum/model counts, schema coverage, credential routes, authentication, billing, and prior push binding. Findings below distinguish existing behavior from required work. See the [implementation plan](implementation-plan.md) for delivery slices.

## Product boundary

The app owns its navigation, conversation timeline, composer, tool cards, questions, approvals, expert roster, connection forms, purchases, and notifications. A system browser can complete OAuth. An isolated content viewer can render HTML/React artifacts. Neither is a substitute for the native product screens.

Build, Library, organization administration, and email/password/username/profile editing are out of scope. Tools may still create or run workflows: their results and user actions must work in the conversation without navigating to those excluded pages.

## Feature coverage

| Capability                   | Existing source and behavior                                                                              | Native acceptance                                                                                                                                   | Server work to evaluate in PR 1                                                                                                                                           |
| ---------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cloud or own server          | Earlier agreed product requirement; auth origin and backend origin can differ                             | Remember the selected server; show which account/server is active; isolate caches, pending auth, credentials, and notification bindings on switches | Versioned discovery at the entered server origin, validated service locations, supported auth/purchase/push/card capabilities, clear unsupported-server response          |
| Email sign-in                | Better Auth email/password and account policies in `frontend/src/lib/auth/auth.ts`                        | Native form, verification/blocked/rate-limit errors, secure session persistence, expiry/renewal, sign-out                                           | Explicit native session/token flow using the existing identity store; retain signup, verification, provisioning and team-login policy; authoritative revocation semantics |
| Google sign-in               | Existing configured Google provider and browser auth routes                                               | System auth browser returns to the initiating app/server/account; cancellation and retry recover                                                    | Bound one-use PKCE/state handoff, expiry and allowlisted callbacks; server-only Next.js auth changes belong in PR 1                                                       |
| Chat with Otto               | `/api/chat/sessions`, session detail, stream POST, history paging                                         | Native timeline/composer, send/stop, history, pending follow-ups, attachments, errors and access limits                                             | Pending-message identity is preserved through admission, drain and history; publish the normalized history contract where needed                                          |
| Chat with experts            | Session `expert_id`, expert APIs, kickoff and existing conversations                                      | Choose an expert, open the correct conversation, preserve expert identity/history, hire/configure through supported cards                           | Reuse ownership-checked APIs; audit expert/action coverage and supported discovery capabilities                                                                           |
| Stream and lifecycle         | Cursor-resumable SSE and persisted rows                                                                   | No duplicated/lost text or actions when backgrounding, reconnecting, relaunching, or receiving a stale checkpoint                                   | Publish exact stream/history schema and compatible recovery rules; preserve message idempotency                                                                           |
| Every tool card              | 89 response enum values; 88 have producing models; 90 model declarations; tool-name dispatch also matters | Native rendering and actionable behavior for every known produced type, legacy tool names and raw SDK outputs                                       | Complete versioned schema/renderer-action registry and compatibility fixtures; current OpenAPI union exposes only 35 models                                               |
| Questions and expert intake  | Clarification cards; `pending_question`; ordinary answer messages today                                   | Single/multiple/custom answers, resume after relaunch, answered/stale state, no duplicate submissions                                               | Stable answer identity and stale-question/retry behavior must be specified; a repeated plain message is not inherently safe                                               |
| Approvals and held calls     | `/api/review/session/{id}`, `/api/review/pending`, `/api/review/action`; held-call records                | Show exact subject, fields, references and spend; edit/approve/decline; reconcile races from other devices                                          | Preserve ownership and server-authoritative status; bind actions to persisted reviews, never inferred model text                                                          |
| New connections              | Provider metadata, OAuth, API-key, device-auth and MCP routes                                             | Native provider/form selection, consent in system browser, return to interrupted chat action                                                        | Mobile-safe OAuth handoff; schema-backed forms; validated server/account/provider binding                                                                                 |
| Existing connections         | List/get/delete, OAuth reconnect and scope upgrade exist                                                  | Rename/edit supported fields, rotate keys, reconnect scopes, choose among multiple accounts, revoke; HubSpot is an acceptance scenario              | A general credential-edit route was not found in `integrations/router.py`; verify other routes or add an ownership-checked update/rotation contract preserving references |
| Provider refusal             | Structured provider-failure event and session connection switch                                           | Explain the refusal and offer valid reconnection/switch actions while retaining the conversation                                                    | Reuse live provider/connection capability data; preserve safe retry semantics                                                                                             |
| Artifacts                    | Workspace upload/list/download/preview and references in rows/text/tool outputs                           | View images, video, PDF, CSV, HTML, React, Markdown, JSON, code and text; download/share other files                                                | Audit authenticated fetch, file limits, MIME handling and untrusted-content isolation; keep account tokens out of artifact execution                                      |
| Runs, schedules and routines | Execution, schedule, routine and trigger APIs/tool responses                                              | Native status/detail and configure/pause/resume/delete sheets reached from cards                                                                    | Replace web-only action hrefs with structured targets; actions must not require Build/Library pages                                                                       |
| Subscribing                  | Stripe status/checkout/manage APIs; one effective `User.subscriptionTier` field                           | Existing subscribers retain access; native plan/purchase/restore/pending/cancel states; no duplicate purchase prompt                                | App Store support and provider-aware entitlement reconciliation are new work; plan source, event ordering, refund/renewal, account binding, and management destination    |
| Notifications                | Current dev has Web Push and completion events; prior experiment has native transport/auth binding        | Permission at a useful moment, register/refresh/unregister, background/terminated taps, requests answered elsewhere clear                           | Native subscription records and APNs transport; preserve authoritative session binding/revocation; persisted attention events and structured targets                      |

## Card coverage is a checked inventory

The [card catalog](card-catalog.md) enumerates every `ResponseType` value and its model declarations. It is not enough to copy the current OpenAPI response union or to claim all raw JSON is supported.

The native contract must cover three sources together:

1. Typed tool payloads: response type plus tool name/variant. `mcp_tool_output` represents both MCP results and sub-session results; `agent_builder_guide` also has two model declarations.
2. Conversation parts outside the ordinary tool chain: expert intake/changes, held-call outcomes, compaction, attachments, and expert-run metadata.
3. SDK and historical outputs: strings, file diffs, older tool names and outputs that do not use the current enum.

There are **25** stream enum values in `backend/copilot/response_model.py`, including comment-only heartbeat/usage and deprecated cursor behavior. Stream frames, comments, checkpoint digests, history reseeding and terminal markers need separate coverage. The web reconnect implementation in `frontend/src/app/(platform)/copilot/stream/turnRuntime.ts` is reference behavior, not code the app should embed in a WebView.

Response model `type` fields are generally `ResponseType` with defaults, not narrow `Literal` declarations. The existing union is not by itself an unambiguous generated native decoder. The contract design must resolve this explicitly.

No `render_ui`/OpenUI producer was found in the inspected baseline. An unmerged integration must not be represented as current support. Future unknown cards need a bounded, readable fallback and an explicit unavailable-action state; new producers must add registry coverage and E2E fixtures before the compatibility check passes.

## Artifact coverage

| Kind                       | Required native behavior                                                      | Boundary proof                                                                                                      |
| -------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Image, video, PDF          | Native preview/player, zoom or paging as appropriate, share/download          | Authorized fetch, denied/deleted file, large file, cancellation and account switch                                  |
| Markdown, text, JSON, code | Readable native text/code, copy/share, safe link handling                     | Long content, malformed data, Unicode, no interpretation as privileged commands                                     |
| CSV                        | Bounded table with accessible headers, horizontal scrolling, export           | Large/ragged cells, Unicode, no implicit formula execution                                                          |
| HTML, React                | Isolated artifact viewer with an explicit content boundary                    | No access to app credentials/native APIs; controlled navigation, network and permissions; app chrome remains native |
| Unknown/binary             | File metadata and explicit download/share                                     | No silent execution; unsupported format remains recoverable                                                         |
| Expert/computer references | Native expert details; explicit remote-session/content viewer where supported | Validate target/access; do not trust arbitrary output URLs as authenticated destinations                            |

## Authentication and subscriptions

Better Auth currently maintains longer-lived cookie sessions and issues one-hour backend JWTs. Native renewal and revocation need an explicit contract. Avoid a parallel account system or migrating authentication into a new service solely for mobile.

`backend/data/credit.py:set_subscription_tier` writes the effective tier directly, and Stripe webhooks invoke it. App Store transactions need recorded provenance and a single reconciliation policy so one provider cannot remove another provider's valid entitlement. Purchase/restore identity, duplicate/out-of-order events, refunds, renewals, pending transactions and existing Stripe subscriptions are required E2E cases.

The simulator experiment can begin with Google/email. Before App Store distribution, resolve the qualifying alternative-login requirement, purchase/storefront rules, and account deletion if account creation is offered. Account deletion is separate from excluded profile editing. These are release decisions, not claims that Apple setup or approval exists. [Apple review guidelines](https://developer.apple.com/app-store/review/guidelines/).

StoreKit's [appAccountToken](https://developer.apple.com/documentation/storekit/transaction/appaccounttoken) links a transaction to a service account. Real purchase settlement requires signed-transaction verification, [App Store Server Notifications V2](https://developer.apple.com/documentation/appstoreservernotifications), product configuration and [sandbox validation](https://developer.apple.com/documentation/storekit/testing-in-app-purchases-with-sandbox). Local purchase fixtures are separate evidence.

## E2E acceptance

Each capability needs a native simulator journey through the production auth/API/persistence code, with deterministic adapters at external model/provider boundaries. UI-only fixture tests remain useful but do not replace that journey. Google/HubSpot provider redirects, real APNs delivery and App Store sandbox settlement also need explicitly labeled provider validation.

Simulator E2E belongs in the new PR gate. The target is a measured critical path below ten minutes on standard free runners, including cold-cache evidence; setting a timeout or moving the tests to nightly does not meet that target. Existing repository examples include four-way frontend sharding and versioned seed/build caches. [GitHub standard runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

## Source anchors

- Identity: `frontend/src/lib/auth/auth.ts`, `frontend/src/app/api/auth/login/with-provider/route.ts`, `autogpt_libs/autogpt_libs/auth/jwt_utils.py`.
- Conversations: `backend/backend/api/features/chat/routes.py`, `backend/backend/copilot/model.py`, `backend/backend/copilot/response_model.py`, `backend/backend/copilot/stream_registry.py`.
- Tool presentation: `backend/backend/copilot/tools/models.py`, `frontend/src/app/(platform)/copilot/components/ToolChain/ToolResult.tsx`, chat history conversion helpers and non-chain message parts.
- Actions: `backend/backend/api/features/graph_executions/review/`, `backend/backend/api/features/integrations/`, `backend/backend/copilot/tools/ask_question.py`, `backend/backend/copilot/gate/`.
- Files: `backend/backend/api/features/workspace/`, `frontend/src/app/(platform)/copilot/components/ArtifactPanel/`.
- Billing: `backend/backend/api/features/billing/subscriptions/routes.py`, `backend/backend/data/credit.py`, `backend/schema.prisma`.
- Notifications: `backend/backend/api/features/push/`, `backend/backend/api/features/home/`, and selectively reviewed native transport/session work on retired branch `codex/mobile-friendly-autogpt`.

Paths in this section are relative to `autogpt_platform/`. The implementation plan records the delivery order and the remaining measured framework/CI gate.
