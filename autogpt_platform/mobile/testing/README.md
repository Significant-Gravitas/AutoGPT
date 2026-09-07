# Native integration fixture

A dependency-free local server for exercising the iOS and Android shells without credentials or a running AutoGPT backend. Every page is labelled **Native integration fixture**. Its screenshots are integration evidence, not screenshots of a live AutoGPT conversation.

Use Node.js 22 or newer. From the repository root:

```sh
node --test autogpt_platform/mobile/testing/server.test.mjs
node autogpt_platform/mobile/testing/server.mjs
```

The server listens on `http://127.0.0.1:8765`; `/copilot` is its entry page and `/health` confirms readiness. Configure the native app's **debug-only origin override** to `http://127.0.0.1:8765` using its platform README. The iOS simulator can access the Mac's loopback address directly. For a connected Android device, `adb reverse tcp:8765 tcp:8765` exposes the same address without opening this fixture to the network. Run `adb reverse --remove tcp:8765` after testing. A separate Android emulator can also use its host alias `10.0.2.2` if the debug origin policy supports it.

The native production default remains `https://platform.agpt.co`. This fixture does not change application configuration.

## Device checks

| Probe                                                      | What to verify on the native app                                                                                                                        |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Native Sign in                                             | System browser opens the fixture consent page; **Connect AutoGPT** returns to the same app; session probe becomes connected.                            |
| Cookie persistence                                         | Reload, terminate, and reopen the app while this server remains running. The session stays connected. The page cannot read the HttpOnly session cookie. |
| Text input                                                 | Type, select, and paste text; dismiss the keyboard; rotate the device; verify controls remain reachable around keyboard and safe areas.                 |
| Single and multiple attachments                            | Open the system picker, select files, then cancel a second selection. Filenames appear on the page; no file bytes are uploaded.                         |
| Same-origin navigation and redirect                        | Stay in the app; native Back returns to the previous page.                                                                                              |
| External links, redirects, and popups                      | Follow the native external-link policy without turning the primary chat webview into an arbitrary browser.                                              |
| File, data, intent, deceptive-host, invalid callback links | Do not grant trusted-origin behavior or establish an authenticated session.                                                                             |
| Attachment download                                        | Show the intended native download/share behavior. Compare local HTTP and the optional trusted HTTPS fixture below.                                      |
| Slow page                                                  | Loading feedback remains visible until the 2.5-second response arrives.                                                                                 |
| HTTP 503                                                   | A received HTTP error page remains distinguishable from a network failure.                                                                              |
| Disconnected transport                                     | Show native recovery UI. Return to the home route or retry after restoring a reachable URL.                                                             |
| Full offline recovery                                      | Stop the fixture, reload, restart it, and retry. Restarting clears fixture sessions, so repeat Sign in afterward.                                       |

For progress evidence, record device name, OS/runtime, app commit, fixture origin, and the probe performed. Keep this page's fixture label visible in screenshots. Verify real hosted chat and production authentication separately.

For an already booted iOS simulator:

```sh
xcrun simctl list devices booted
xcrun simctl io booted screenshot /private/tmp/autogpt-native-fixture.png
```

When multiple simulators are booted, replace `booted` with the intended simulator's UDID. Native UI automation can use Cua Driver or the available computer-use tools against the Simulator app; observe fresh UI state before acting and verify the result afterward.

The generated-blob probe follows the web chat export sequence exactly: create an object URL, append and click an anchor with `download`, remove the anchor, then immediately revoke the URL. Verify the native share sheet receives `native-fixture-generated.md`.

## Browser authentication contract

The fixture mirrors the native handoff shape:

1. Native opens `GET /api/auth/mobile/start?code_challenge=<S256>&state=<state>`. The challenge is 43 base64url characters; state is 32–128 base64url characters.
2. Start redirects to `/auth/mobile` with the same parameters. The fixture skips real account authentication and displays explicit browser consent.
3. **Connect AutoGPT** sends same-origin `POST /api/auth/mobile/authorize` with JSON `{code_challenge,state}`. It receives `{url:"autogpt://auth/callback?code=...&state=..."}` and navigates there.
4. Native validates its pending callback and sends `POST /api/auth/mobile/exchange` with JSON `{code,code_verifier}` and an `Origin` header exactly matching its configured origin. Redirects should remain disabled for this exchange.
5. Exchange responds `200 {"success":true}` with a persistent HttpOnly, SameSite=Lax `better-auth.session_token` cookie (`__Secure-better-auth.session_token` over HTTPS). A second HttpOnly `session_data` cookie with an `Expires` date exercises multiple `Set-Cookie` headers. Native copies the cookies into its webview's persistent cookie store and loads `/copilot` itself.

Codes expire after 60 seconds and are consumed on successful exchange. The in-memory fixture session expires after a day or a server restart. The fixture tests exercise these mechanics; they do not replace the production auth implementation's authorization, revocation, source-session, or security tests.

## Optional HTTPS download check

HTTP loopback is useful for native debugging but does not prove production HTTPS download behavior. To serve the same fixture over TLS, provide an existing certificate and private key trusted by the test device:

```sh
node autogpt_platform/mobile/testing/server.mjs \
  --port 8766 \
  --tls-cert /absolute/path/localhost-cert.pem \
  --tls-key /absolute/path/localhost-key.pem
```

Set the native debug origin to `https://localhost:8766`, matching the certificate's subject names, and repeat the attachment probes. HTTPS sessions add the Secure cookie attribute. Do not disable certificate verification in the app. This command neither creates nor installs certificates; keep private keys outside the repository.

The default server binds only to loopback. `--host` exists for an explicitly chosen local device setup; this fixture has no real account authentication and should never be deployed publicly.
