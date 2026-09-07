# AutoGPT for Android

A small native host for the existing AutoGPT website. Chat rendering, streaming, conversation history, agents, tools, and account screens remain in the web application. The app defaults to `https://platform.agpt.co/copilot`.

The primary device target is Pixel 11 Pro, the compact flagship Android peer of iPhone 17 Pro. This project compiles against Android API 36, targets API 36, and supports Android 10/API 29 or later with an updated Android System WebView. No emulator image or Android Studio installation is required to build it.

## Build and install

Requirements:

- Java 17.
- Android SDK platform 36, build tools 35.0.0, and platform tools.
- `ANDROID_HOME` pointing to that SDK. Alternatively, put `sdk.dir=/absolute/path/to/sdk` in the ignored `local.properties` file.

From this directory:

```sh
./gradlew --no-daemon assembleDebug testDebugUnitTest lint
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The Gradle wrapper pins Gradle 8.13 and verifies its distribution SHA-256. Android Gradle Plugin 8.13.2 and Kotlin 2.2.21 are pinned. Builds use at most two Gradle workers and a 1.5 GiB Java heap. The Gradle/AndroidX versions are deliberately pinned to the API 36 toolchain; lint may report newer available dependencies.

The debug APK uses the normal local Android debug signing key. It is suitable for development and device testing. Production signing, Play distribution, and release credentials are not configured.

## Connect to a server

Open the app menu → **Server settings**, enter a bare HTTPS origin, and confirm the destination. Examples are `https://platform.agpt.co` and `https://my-autogpt.example:8443`. Paths, credentials, query strings, fragments, and ambiguous hostnames are rejected. Changing the server clears the app's existing website data before opening the new server.

Debug builds additionally allow HTTP only on `localhost`, `127.0.0.1`, and the Android emulator host address `10.0.2.2`. For the mobile fixture running on port 8765:

- Emulator: configure `http://10.0.2.2:8765`.
- USB device: run `adb reverse tcp:8765 tcp:8765`, then configure `http://127.0.0.1:8765`.

The fixture is a labeled test website, not evidence that production authentication or real chat APIs work. The new mobile authentication routes must be deployed on a chosen AutoGPT server before browser sign-in works there. They are not assumed to exist on the public production server yet.

## Browser sign-in

The app's **Sign in** action opens a system Custom Tab at `/api/auth/mobile/start` with a fresh S256 PKCE challenge and random state. A matching `autogpt://auth/callback` exchanges the single-use code at `/api/auth/mobile/exchange`. Redirects are disabled for that HTTP exchange; requests include the exact configured `Origin` header.

Successful exchanges must return HTTP 200 and a usable HttpOnly BetterAuth session-token cookie, alongside any session-cache cookies. The native client ignores response bodies and enforces an absolute 30-second exchange deadline; cancellation disconnects the owned HTTP request. HTTPS session cookies must also be Secure. Explicit cookie domains must exactly match the configured host; accepted domains are removed before inserting host-only WebView cookies. Before installing the complete fresh cookie family, the app destroys the old page and waits for `WebStorageCompat.deleteBrowsingData` to finish. This removes stale session-cache cookies, local storage, IndexedDB, and cache. Server changes and account replacement are serialized through a retained ViewModel, including activity recreation.

PKCE verifiers and pending cookies stay in memory and survive rotation. They are never placed in URLs, preferences, saved-instance bundles, logs, or a credential store. If Android kills the whole app during sign-in, the callback is rejected and the user starts again. Pending sign-in expires after ten minutes. A state-bound `access_denied` response cancels without clearing the previously signed-in account.

An updated System WebView with the `DELETE_BROWSING_DATA` feature is required for signing in, switching servers, and clearing the session. Older WebViews display an update message. App backup and device-transfer backup are disabled for website/session data.

## Files and navigation

Web file inputs use Android's document picker without broad storage permissions. Each selection is bound to its original WebView, document generation, and server; late results after navigation or recreation are discarded. The native activity validates content-URI shape without opening cloud files on the main thread, leaving file reads to WebView. The existing web voice recorder can request Android microphone permission; only audio capture from the configured origin in the current WebView is eligible. Camera and unrelated web permissions remain denied, and pending microphone grants are discarded after navigation or account/server changes. HTTPS downloads from the configured origin can be saved to temporary app storage and exported through the system share sheet. Such requests carry cookies only to the same origin, including every redirect, and are limited to 50 MiB. Temporary exports are pruned after a day or beyond a 100 MiB retained budget and cleared during account/server changes. Active requests are cancelled when their WebView is destroyed; they cannot present old-account share results after a switch.

Generated chat files use the web `saveBlob` helper and the narrow `AutoGPTDownloads` WebMessage interface. The object exists only when the WebView supports origin-scoped web messages. The native receiver verifies the configured source origin, the current top-level page, and `isMainFrame` for every request. It offers only a write-only download flow:

1. `{type:"start", id, filename, mimeType, size}` opens Android's **Create document** picker. `ready` is sent after a destination is selected and private staging storage is opened. The chosen destination is untouched while chunks arrive.
2. `{type:"chunk", id, index, data}` streams standard base64 bytes into an app-owned temporary file and receives `{type:"ack", id, index}` after the write completes.
3. `{type:"finish", id}` succeeds only when exactly the declared number of bytes has been written, copies that completed file to the chosen destination, then replies `complete`.
4. `{type:"cancel", id}` replies `cancelled`. Errors reply `error` with a short message.

There is one transfer at a time, a 50 MiB file limit, a 64 KiB decoded-chunk limit, and a 90 KiB JSON-message limit. Chunks must be sequential and acknowledgement-paced. Inactivity for two minutes, full-page navigation, server changes, activity destruction, or renderer death aborts the transfer and removes app-owned staging. Selected document URIs are never automatically deleted: Android's Save picker can return an existing file after overwrite confirmation. Cancellation before the final copy leaves the selected destination untouched; interruption during that copy can leave partial output, which the app reports. Provider writes use a separate pool with at most two active copies, cancellable descriptors, and a cancellation signal. An unresponsive provider or kernel operation cannot be forcibly terminated; exhausted copy slots return a busy error without opening more provider handles. The interface exposes no native reads, cookies, arbitrary file paths, or commands. No DOM or browser-API monkeypatches are injected.

Same-origin links stay in the WebView. HTTPS video embeds and configured-origin blob previews remain available in subframes, which never receive native download privileges. External HTTPS links and user-initiated popups go to the system browser; `mailto:` and `tel:` links use their normal handlers. Invalid or unsolicited external navigation is blocked. Certificate errors always fail closed. Android Back follows web history and returns to the operating system at the root. The shell handles system-bar/cutout insets once, forwards keyboard insets to modern WebView, restores bounded navigation history, and presents recovery after renderer termination.

## Formatting

All Kotlin sources and Gradle Kotlin scripts use ktfmt 0.64 with Kotlin language style. With the [official formatter](https://github.com/Kotlin/ktfmt/releases/tag/v0.64) downloaded, format the entire project from this directory:

```sh
rg --files -g '*.kt' -g '*.kts' -0 | xargs -0 java -jar /path/to/ktfmt.jar --kotlinlang-style
```

The launcher image reuses the web app's notification icon. The themed monochrome launcher uses the existing `AutoGPTLogoWhite` SVG geometry, without its wordmark. Native status screens support system light/dark appearance, large text, and scrolling when the available height is short.

## Verification

Automated checks cover exact-origin matching and spoof attempts, debug HTTP restrictions, the RFC 7636 challenge vector, callback state/expiry/cancellation, session-cookie protection, download message bounds, untrusted-frame rejection, exact byte counts, chunk ordering, and cancellation. Android lint and the debug APK build run for the entire project.

Device verification is still required for real browser sign-in and switching between two accounts, Custom Tabs callbacks, file pickers and cloud document providers, portrait/landscape keyboard behavior, predictive Back, background process death, TalkBack, large text, and real streaming chats. An APK build and JVM tests do not substitute for those runtime checks.
