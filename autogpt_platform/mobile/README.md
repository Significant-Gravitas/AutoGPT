# AutoGPT mobile apps

Native iOS and Android hosts for AutoGPT's existing chat interfaces. The default entry point is `https://platform.agpt.co/copilot`; navigation to other AutoGPT pages stays on the same configured website. Conversations, streaming, agents, tools, history, account settings, and onboarding remain implemented by the web application. Web improvements reach the apps without copying components or issuing an app update.

- [Build and test iOS](ios/README.md)
- [Build and test Android](android/README.md)
- [Native integration fixture and PostgreSQL checks](testing/README.md)
- [Progress screenshots](screenshots/README.md)

This is a development prototype. Store publication, production deployment, release signing identities, and store privacy declarations are not configured by this change.

## Native responsibilities

The shells manage persistent website sessions, safe URL handling, system-browser sign-in, keyboard and safe areas, Back navigation, loading/recovery, file selection, file export, and user-controlled microphone permission. General responsive-web design remains in the separate web workstream.

The iOS app uses UIKit, WebKit, AuthenticationServices, and a local Swift package. Android uses the platform WebView and small AndroidX components. There is no separate chat API client, copied message renderer, or React Native/Capacitor runtime.

HTTPS origins can be selected explicitly for self-hosted and preview deployments. Changing servers clears the app's website session. Debug builds allow loopback HTTP for local tests; release builds require HTTPS. Only the chosen origin is trusted inside the main WebView. External web links use the system browser, and unsupported schemes are rejected.

## Browser sign-in

The selected web deployment must include the mobile authentication endpoints from this PR. Until those endpoints are deployed, the existing hosted website can be used through **Open in browser**, but the app's browser-to-app sign-in cannot complete against that deployment.

1. The app creates a random proof and state, then opens the existing web login in the system authentication browser.
2. The signed-in browser asks the user to connect the app, displaying and binding the current account.
3. The website returns a 90-second, single-use code through the fixed `autogpt://auth/callback` URL.
4. The app validates its pending state and exchanges the code with its proof. The server rechecks account policy and source-session validity and creates a fresh session.
5. The app clears previous website identity/cache data, installs the new HttpOnly cookies, and loads the hosted chat.

Long-lived session tokens and provider credentials are never put in the callback URL. The browser and app retain separate sessions. Ticket storage reuses Better Auth's existing verification table; no database migration or new package is required.

## Files and voice

File inputs use the operating system's picker without broad storage permissions. iOS uses WebKit's download support, including generated blobs, followed by the system share/export sheet.

Android cannot download browser `blob:` URLs directly. The existing web export functions use one shared `saveBlob` helper that detects a narrow, origin-scoped `AutoGPTDownloads` capability. It asks the user where to save, transfers at most 50 MiB in acknowledged chunks, and stages bytes privately before copying the completed file. It cannot read native files, expose cookies, or invoke arbitrary native methods. Other browsers and iOS retain the normal browser download path. Selected Android document URIs are never automatically deleted on cancellation; provider failures can leave partial output at the user-selected destination.

Microphone requests are restricted to the current trusted page and require platform/user permission. The fixture's microphone probe stops tracks immediately and records or uploads no audio.

## Maintenance and verification

Keep `/api/auth/mobile/*`, the fixed callback, and the `AutoGPTDownloads` message contract backward compatible when changing the web application. The [platform build instructions](android/README.md#files-and-navigation) document the Android protocol and limits. App packages still need occasional operating-system, security, signing, and dependency maintenance; hosting the UI removes the separate chat-feature implementation.

The mobile CI workflow builds the native projects and runs their policy/fixture tests. Monthly Dependabot checks cover Android dependencies, grouping minor and patch updates to keep the review queue small; major updates remain separate. Dependency updates still require passing CI and review. Existing frontend CI covers the shared web changes. The optional PostgreSQL harness uses an isolated disposable container and multiple worker processes to verify one-use redemption and session policy against the real database adapter.

Current evidence is recorded separately for compilation, policy tests, native UI checks, and real hosted behavior. Local fixtures are labelled explicitly and do not prove a live account, model response, provider login, or Android device behavior. Target devices are iPhone 17 Pro and Pixel 11 Pro; the initially available local iOS runtime is iPhone 16 Pro / iOS 18.3.
