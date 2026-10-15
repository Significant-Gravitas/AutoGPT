# AutoGPT SwiftUI client

The SwiftUI fallback for the iOS experiment, built after the React Native cold builds missed the ten-minute gate. Scope, gates and evidence rules are in the [implementation plan](../../../docs/engineering/native-mobile/implementation-plan.md). The native launch journey and the local production-server email/chat/relaunch journey now pass on the shared iPhone 17 simulator. This is still a partial experiment: full card actions, provider integrations, free-runner timing and independent app reviews remain open.

## Build

```sh
cd autogpt_platform/mobile/apple
xcodebuild -project AutoGPT.xcodeproj -scheme AutoGPT -configuration Debug \
  -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath .build/SwiftUISpike -jobs 3 ARCHS=arm64 ONLY_ACTIVE_ARCH=YES \
  CODE_SIGN_IDENTITY=- COMPILER_INDEX_STORE_ENABLE=NO -showBuildTimingSummary build
```

The app is `.build/SwiftUISpike/Build/Products/Debug-iphonesimulator/AutoGPT.app`. `project.yml` is the XcodeGen spec; the generated `AutoGPT.xcodeproj` is committed, so CI needs no generator. Run `xcodegen generate` after changing the spec or the file list. Swift is formatted and linted with the toolchain's `swift-format` and `.swift-format`; see [AGENTS.md](AGENTS.md).

Scheme `AutoGPT`, iOS 17+, iPhone, bundle `com.agpt.mobile.experimental`, Swift 6 language mode. Ad hoc simulator signing; signing stays enabled because the simulator Keychain needs the signed identity. System frameworks only, all linked: Foundation, SwiftUI, UIKit, Security, CryptoKit, AuthenticationServices, StoreKit, UserNotifications, WebKit, AVKit, PDFKit and QuickLook.

## Validation evidence

Local Mac, Xcode 27.0 and iOS 27, 2026-10-12:

- A fresh DerivedData build of optimized Release passed in 19.87 seconds; Debug passed in 10.03 seconds afterward. Existing OS/toolchain caches were not cleared. These are local build timings, not CI or whole-journey measurements.
- The native server selector and nonlocal HTTP rejection journey passed in 22.6 seconds.
- The native interaction fixture journey passed in 71 seconds, covering secure storage, long streaming, question answers, connection edit rejection and isolated HTML. The keyboard and switch interactions were reproduced failing before correction.
- Email sign-in, a streamed Otto reply, hard relaunch and restored history passed through the production Better Auth/API/executor/persistence stack in an isolated local workspace. A separate API check verified exactly one user message and one assistant reply. Only the external model was scripted.
- The first standard-runner SwiftUI job compiled Release and Debug under Xcode 26.6, then exceeded ten minutes during the Release launch journey after a 5m39s simulator boot. Debug interactions were skipped. The workflow now tries the standard Intel runner with universal simulator builds and boot/test resource sampling; the full cold/warm timing gate and iOS 17 runtime remain unverified. See the [build evidence](../README.md#checks-and-build-evidence).

Revalidated on October 14 with Xcode 27 / iPhone 17: Debug and Release compile, Release onboarding and the production-server email/chat/relaunch journey pass, including the separate persisted-row assertions. The current Debug interaction journey also passes after correcting its Maestro swipe command. Manual large-text checking verifies that the introduction remains scrollable above the keyboard. These checks do not close the full accessibility, provider or standard-runner gates.

## Design system mapping

Use native navigation stacks, tab bars, sheets, menus, forms, system typography, Dynamic Type and keyboard behavior. AutoGPT's current `frontend/src/components/atoms/Button/helpers.ts` supplies the charcoal primary color (`zinc-800`, `#27272a`), with a light neutral accent in dark appearance. Message surfaces use zinc-100/zinc-800 with zinc-200/zinc-700 borders. Question choices share that neutral palette, add a visible selection border and keep native checkmark accessories. Primary actions use the rounded capsule shape from the web button atom, while navigation, forms and sheets retain system behavior.

Bundled artwork is converted losslessly from the same sources named by `frontend/src/components/molecules/AutopilotAvatar/helpers.ts`: `public/experts/transparent/otto.webp` for welcome/empty states, and `public/autogpt-characters/v1.1/otto/neutral/256.webp` for portraits. Bundling keeps Otto visible before sign-in and on a self-hosted server without those static assets. Expert portraits use the no-cookie, no-redirect transport with a four-MiB limit and bounded thumbnail decoding.

The welcome screen gives Cloud sign-in a primary action and keeps self-hosting alongside it. Conversation introductions scroll when the keyboard or large accessibility text leaves less room, while the composer stays above the keyboard.

## What is implemented

**Ported from the reference client** (`../src/`), with the same protocol rules:

- Cloud or own server. Origins are validated as in the reference: bare origins only, HTTPS except loopback HTTP, no credentials in addresses.
- Email sign-in. Only the signed `set-auth-token` header is accepted, never the JSON body token. The new session is verified against the token exchange and an API probe before it is stored.
- Session storage. The Keychain item is partitioned by server and account, readable only when unlocked on this device, and wiped on a fresh install. API JWTs are memory-only, with one shared exchange. Only the auth server's 401 ends a session; 5xx, 429, 503 and network errors keep it. Every account or server change starts a new generation: the signed-in UI and its stores are rebuilt, old token sources fail immediately, and late callbacks are ignored.
- Transport. No cookie jar, cache or credential store, and redirects are refused. JSON exchanges have one deadline covering connection, headers and body; the reference's body-timeout gap is closed. Streams have a header deadline, 30 s liveness and 25 min rotation. Cancelling a task cancels its URLSession task.
- Conversations. The Otto and expert lists, conversation creation, history paging and the SSE turn log follow the reference row rules. Resume handles cursors, the 409 checkpoint reseed and the 410/404 settle path. Queued turns are polled. Sends use stable client message IDs, explicit Retry/Delete for unconfirmed sends, Stop, and background/foreground resume. Nothing is replayed automatically: failed sends are memory-only.
- Native Chats and Company tabs, the conversation screen, tool step rows with secret-masked, bounded details, and an Account sheet.

**Interaction spike surfaces:**

| Surface                              | Production path                                                                                                                                                       | Fixture route                                                                              |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Long streamed conversation, keyboard | Real controller and server stream; follows new content only while the reader is at the end; interactive keyboard dismissal                                            | Long history (paged) plus a long reply with reasoning, a table, code and a live tool step  |
| Partial text selection               | Every Markdown block and user bubble is a non-editable `UITextView` (SwiftUI `Text` only copies whole strings)                                                        | Same views                                                                                 |
| Question card                        | Rendered for real `ask_question` outputs; answers go through the ordinary send path in the web client's exact wording, only while the asking reply is newest and idle | Single, multiple and free-text questions                                                   |
| Editable connection form             | Real connection list is read-only: the server has no in-place update route                                                                                            | Rename, write-only key replacement, empty-key refusal, provider rejection (`rejected-key`) |
| Isolated artifact viewer             | Not wired to authorized artifact fetches yet                                                                                                                          | HTML isolation self-check, PDF, CSV (formulas shown as text), Quick Look                   |

The artifact `WKWebView` uses a fresh nonpersistent data store, no script message handlers and an `about:blank` origin with no file access. A content security policy and a content blocker stop network loads. Pop-ups, alerts and media permissions are refused. Navigation is blocked; a tapped http(s) link asks before opening in Safari.

**Provider boundaries.** Google sign-in, APNs and App Store purchases are adapters that report they are unavailable; the UI shows an explanation, not a control. `SystemBrowserSession` (ephemeral `ASWebAuthenticationSession`) exists for the future Google and OAuth handoffs but is unreachable until a server advertises one.

## UI-protocol fixture route

Debug builds only (`UI_PROTOCOL_FIXTURES`), opened with the launch argument `-AutoGPTUIProtocolFixtures YES`, for example `xcrun simctl launch <udid> com.agpt.mobile.experimental -AutoGPTUIProtocolFixtures YES`. Every fixture screen has an orange "Fixture — not connected to a server" banner. The fixtures drive the real controller, timeline, cards and viewers with local data and never contact a server. This is **fixture coverage, not integrated tool behavior**. Release builds contain none of it.

## Native journey plan

Run in the shared T3 Device simulator:

1. `../e2e/launch.yaml`, unchanged.
2. `python3 ../scripts/run-chat-journey.py --device <udid> --auth-origin http://127.0.0.1:18800 --api-origin http://127.0.0.1:18806` (drives `../e2e/email-chat.yaml`). Every identifier and string those flows use is preserved: `choose-cloud`, `choose-own-server`, `own-server-auth-url`, `own-server-api-url`, Continue, the validation sentence, `sign-in-email`, `sign-in-password`, `sign-in-submit`, `message-otto`, `composer-input`, `composer-send`, `chat-row-<id>`. Message text is in `UITextView`s; it is exposed as the accessibility value, not the label, and Maestro matches it in the passing journey.
3. [`e2e/ui-protocol-fixtures.yaml`](e2e/ui-protocol-fixtures.yaml), passes the device Keychain round trip, long stream, question answers, rejected connection-key edit and artifact isolation checks. It is UI/protocol fixture coverage, not provider integration.
4. Manual checks for the gate: keyboard over a streaming reply, scrolling up while a reply streams, partial selection in a paragraph, VoiceOver order and labels, the largest accessibility text size, Reduce Motion, light and dark appearance, backgrounding mid-stream and returning, and relaunch with a stored session. Also test sign-out, a revoked session (expect the "session ended" notice) and switching servers.

## Gaps and concerns

- **Not yet proven:** the full interaction/accessibility/provider gate, free-runner timing and the iOS 17 runtime. Xcode 26.6 Release and Debug compilation passed on the standard ARM runner. Short conversations align at the top on iOS 18+; iOS 17 retains the bottom alignment fallback.
- **Keychain on CI:** the Debug native journey writes, reads and deletes a temporary credential through the real Keychain APIs. Its first run on the free runner remains pending; the local production-server sign-in also proved persistence across app termination.
- **Question answers** reuse the ordinary send path. Without a server answer identity and stale-question contract, two devices (or a card reopened elsewhere) can both answer. The client-side guard is only "newest and idle". Draft answers are view state and can be lost if the card scrolls far offscreen.
- **Connections:** no in-place update or rotation route that preserves credential identity, so editing exists only as fixture coverage. OAuth reconnect is described, not offered.
- **Artifacts:** no authorized artifact fetch is wired. React artifacts will need bundled renderer code because the policy blocks remote scripts. The content-blocker callback assumes WebKit completes on the main thread.
- **Selection** works within one block (a paragraph, list item or code block), not across blocks.
- **Cloud** sign-in only works after production issues signed native sessions (PR 1).
- **Not started:** attachments, expert kickoff, every other card family and action, run, schedule and expert sheets, native viewers wired to real files, Google, APNs, purchases and notification routing. Every card other than the question falls back to the read-only details view, which says its actions aren't available.
