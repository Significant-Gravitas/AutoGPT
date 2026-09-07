# AutoGPT for iOS

A small UIKit/WKWebView host for the existing AutoGPT website. Chat, streaming, agents, history, settings, and onboarding stay on the website. The app provides safe navigation, browser sign-in, persistent website storage, keyboard/safe-area integration, native file selection, and attachment export. There is no JavaScript-to-native bridge on iOS.

## Build

Requires Xcode, its iOS platform support, and [XcodeGen](https://github.com/yonaskolb/XcodeGen). The app supports iOS 16 and newer and targets the iPhone 17 Pro form factor; iPad and rotation remain enabled.

From the repository root:

```sh
xcodegen generate --spec autogpt_platform/mobile/ios/project.yml
xcodebuild \
  -project autogpt_platform/mobile/ios/AutoGPT.xcodeproj \
  -scheme AutoGPT \
  -destination 'generic/platform=iOS Simulator' \
  -derivedDataPath /private/tmp/autogpt-mobile-derived \
  CODE_SIGNING_ALLOWED=NO build
swift test --package-path autogpt_platform/mobile/ios
```

Open the generated `AutoGPT.xcodeproj` in Xcode for device development. Choose your own development team for a physical device; no signing identity is committed. Store publication is not configured.

If a newer Xcode has its simulator SDK but only an older simulator runtime, an SDK-only build can still be installed with `simctl` on a compatible older runtime. This does not enable Xcode's normal UI-test destination discovery:

```sh
xcodebuild \
  -project autogpt_platform/mobile/ios/AutoGPT.xcodeproj \
  -target AutoGPT -sdk iphonesimulator -arch arm64 -configuration Debug \
  SYMROOT=/private/tmp/autogpt-mobile-compile \
  CODE_SIGNING_ALLOWED=NO build
xcrun simctl install booted /private/tmp/autogpt-mobile-compile/Debug-iphonesimulator/AutoGPT.app
xcrun simctl launch booted com.agpt.mobile
```

Install the matching Xcode platform support before running the included XCUITest target normally. Use a specific simulator UDID instead of `booted` when more than one device is running.

## Connection and authentication

The default is `https://platform.agpt.co/copilot`. **App menu → Server settings** accepts a deliberate HTTPS origin, including a self-hosted or preview deployment. Paths, embedded credentials, queries, and fragments are rejected. Changing servers clears this app's website data.

The same deployment must include this PR's `/api/auth/mobile/start`, `/api/auth/mobile/authorize`, and `/api/auth/mobile/exchange` endpoints. Until those endpoints are deployed, **Open in browser** can use the existing website, but native browser-to-app sign-in cannot complete against that deployment.

Sign-in uses `ASWebAuthenticationSession` and an S256 proof bound to a random pending state. The browser callback contains only a short-lived one-use code. The app exchanges the code with redirects disabled, clears previous website identity/cache data, and installs the returned HttpOnly cookies into WKWebView. It neither stores provider credentials nor moves cookies through a callback URL. Canceling the browser returns to the sign-in screen.

External user links open in the system browser. Programmatic external navigation asks before opening. Unsupported schemes are blocked. Same-origin web navigation stays in the app. App and system browser sessions remain independent.

## Local checks

Run the [native integration fixture](../testing/README.md). Debug builds permit loopback HTTP addresses. Release builds require HTTPS. Set the fixture through Server settings or use the debug-only launch override:

```sh
SIMCTL_CHILD_AUTOGPT_ORIGIN=http://127.0.0.1:8765 \
  xcrun simctl launch booted com.agpt.mobile
```

Stop an already running copy of the test app before changing its launch environment. The override is ignored in Release builds. The fixture tests browser return, multiple cookie transfer, session persistence, navigation, keyboard, file selection, downloads, and recovery without a live account.

Format and lint all iOS source directories with the Swift formatter bundled with Xcode:

```sh
swift-format format --in-place --recursive \
  autogpt_platform/mobile/ios/App autogpt_platform/mobile/ios/Sources \
  autogpt_platform/mobile/ios/Tests autogpt_platform/mobile/ios/UITests \
  autogpt_platform/mobile/ios/Package.swift
swift-format lint --strict --recursive \
  autogpt_platform/mobile/ios/App autogpt_platform/mobile/ios/Sources \
  autogpt_platform/mobile/ios/Tests autogpt_platform/mobile/ios/UITests \
  autogpt_platform/mobile/ios/Package.swift
```

## Current verification

- Swift origin and PKCE contract tests pass on the host.
- The app builds with the iOS 26.5 simulator SDK.
- It launches on the installed iPhone 16 Pro / iOS 18.3 simulator and intercepts the real hosted site's login redirect.
- The local fixture's system-browser sign-in returns to the app with both token and cache cookies installed.
- Exact iPhone 17 Pro / iOS 26 runtime testing, real-provider sign-in, and release signing remain separate checks.
