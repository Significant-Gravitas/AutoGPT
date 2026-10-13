# AutoGPT mobile shell

The iOS and Android apps in `autogpt_platform/mobile` load the hosted frontend
at `/home` (the current Copilot entry point). Swift and Kotlin handle browser
sign-in, navigation, file selection, and microphone permission. Next.js and
agent execution stay on the server. No wrapper vendor or paid build service is
required.

`.github/workflows/platform-mobile-ci.yml` builds both shells and runs their
native tests on pull requests. It needs no signing keys and does not publish.

This is a development build, not a store-ready release. Deploy the frontend
changes to a test environment before testing login. The production site will
not have the new auth endpoints until those changes ship.

## iOS

Requires macOS, Xcode with the iOS SDK, and XcodeGen (`brew install xcodegen`).
The app supports iOS 16 and later.

```sh
cd autogpt_platform/mobile/ios
xcodegen generate
swift test
xcodebuild -project AutoGPT.xcodeproj -scheme AutoGPT \
  -sdk iphonesimulator -configuration Debug CODE_SIGNING_ALLOWED=NO build
open AutoGPT.xcodeproj
```

Select an installed simulator in Xcode and run, or choose your signing team
and a connected iPhone. Install an iOS simulator runtime through Xcode Settings
if none exists. The generated Xcode project is ignored; edit `project.yml` and
regenerate it. `Info.plist` comes from that specification too.

For a test deployment, set the `AUTOGPT_ORIGIN` build setting to its HTTPS origin
(for example `https://mobile-test.example.com`). On the command line, append
`AUTOGPT_ORIGIN=https://mobile-test.example.com` to `xcodebuild`.
Use an origin with no path or trailing slash. Do not add ATS exceptions.

## Android

Requires JDK 17 and the Android SDK. Open `autogpt_platform/mobile/android` in
Android Studio, or set `ANDROID_HOME` and `JAVA_HOME` for command-line builds.
The app supports Android 8 and later and targets API 36. Gradle downloads the pinned build tools;
the wrapper checks the Gradle distribution checksum.

```sh
cd autogpt_platform/mobile/android
./gradlew :app:assembleDebug :app:testDebugUnitTest :app:lintDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Use `-PautogptOrigin=https://mobile-test.example.com` to build against a test
deployment. Cleartext HTTP is disabled. The debug package is
`co.agpt.autogpt.dev`; the release package is `co.agpt.autogpt`. Both use the
`autogpt` callback scheme, so install only one when testing sign-in.

The launcher icon copies `frontend/public/notification-icon-512.png` unchanged.

## Sign-in protocol

1. The native app generates a 32-byte verifier and a separate random state. It
   opens `/api/auth/mobile/start` in an iOS authentication session or Android
   Custom Tab, sending the SHA-256 verifier challenge and state.
2. The server sends signed-out users through the existing login page. The
   existing email, Google, and other configured provider flows stay in the
   browser, including platform-user provisioning.
3. The user confirms the displayed account. Only a POST from the exact web
   origin with a live session can approve the handoff.
4. The server creates a random code, valid for 60 seconds, in Better Auth's
   existing verification table. Its identifier binds the code hash to the
   verifier challenge; its value references the source session.
5. The fixed `autogpt://auth/callback` link carries the code and state. No
   session token goes into a URL. The native app rejects wrong states, wrong
   callback routes, duplicate parameters, and attempts older than 10 minutes.
6. The app POSTs the code and verifier to `/api/auth/mobile/exchange`. The
   server atomically consumes the grant, checks the source session and ban
   status, and creates a separate app session. The webview receives HttpOnly
   cookies. Android exchanges over HTTPS and imports response cookies; iOS
   POSTs inside WKWebView. Both then load `/home`.

These routes live in the existing self-authenticating `/api/auth/*` handler;
they do not add protected frontend pages or require a schema migration.
The tests run the real Better Auth handler with its in-memory adapter and cover
consent, origin checks, verifier binding, expiry, concurrent replay, revocation,
and bans. Production uses Better Auth's database-backed atomic consume.

Keep request bodies and full auth callback URLs out of logs. Mobile grant
records expire through the same mechanism as other Better Auth verifications.
Do not enable a secondary verification cache without atomic `getAndDelete`.

## Navigation and permissions

The main webview stays on the exact configured HTTPS origin. Other HTTP(S)
links open in the browser. No JavaScript-to-native bridge exposes native
commands. Android refuses cleartext traffic, mixed content, third-party
cookies, and local file access. The release build disables webview debugging
and app backup.

The apps retain webview cookies and avoid reloading on every foreground event,
which would discard chat drafts. The web app owns stream reconnection. iOS
supports swipe-back, native uploads, and download sharing. Android supports the
system back button, file selection, microphone requests, and keyboard insets.
Android HTTPS downloads open in the browser; blob downloads need the browser
version for now.

Integration OAuth popups open in the browser. Each provider needs a device
test: some existing flows poll the server, while others may still depend on
`window.opener`. Do not claim all integrations work before testing them.

## Device acceptance checks

Against a test deployment, on both iOS and Android:

- Sign in with email and Google; cancel and retry; verify account switching.
- Confirm session persistence after relaunch and expiry after sign-out.
- Start a Copilot run, stream its reply, attach a file, and view its result.
- Background during a run, change networks, and resume without duplicate runs.
- Test keyboard overlap, back navigation, rotation, and offline retry.
- Grant and deny microphone permission; select and cancel file uploads.
- Test an integration connection, an external link, and a downloadable result.
- Verify that an unsolicited or replayed callback cannot sign the app in.

Automated builds and handler tests do not replace these device checks.

## Work before store submission

- Native push: APNs/FCM credentials, user/device token registration and removal,
  backend delivery, notification permission, and links into the correct run.
- Billing: choose the approved mobile purchase model; adapt Stripe links and
  paywalls or implement store billing and purchase reconciliation. The shell
  currently exposes the hosted site's billing UI without changes.
- Review social-login rules, account deletion, privacy disclosures, AI-content
  reporting, permissions, and store review access for the chosen scope.
- Add final iOS app icons, Android adaptive icons, screenshots, age ratings,
  support links, and crash reporting. Audit SDK privacy declarations.
- Test on actual devices and current OS versions; confirm store SDK requirements.
- Set the final bundle/package IDs and versions. Use AutoGPT's organization
  accounts and keep signing keys outside the repository.

For iOS, archive the Release scheme with the team's signing settings, then
upload through Xcode Organizer for TestFlight and App Store review. For
Android, use Android Studio's Generate Signed Bundle flow with the team's
upload key, then upload the AAB to a Play internal testing track. Do not use
the debug APK for a store release. Publishing and signing are separate from
the local build; no store upload happens automatically.
