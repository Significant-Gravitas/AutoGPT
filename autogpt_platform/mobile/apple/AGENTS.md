# SwiftUI client

Read `../AGENTS.md`, the platform guides and `docs/engineering/native-mobile/implementation-plan.md` first. This folder is the iOS client for the native AutoGPT experiment. The React Native source in `../src/` is reference material during the spike.

## Project

- `project.yml` is the XcodeGen specification; `AutoGPT.xcodeproj` is generated from it and committed so CI never installs XcodeGen. After changing `project.yml` or adding, moving or removing files, run `xcodegen generate` here and commit both.
- Scheme `AutoGPT`, iOS 17 deployment target, iPhone only, bundle `com.agpt.mobile.experimental`, Swift 6 language mode with complete concurrency checking.
- Simulator builds are ad hoc signed (`CODE_SIGN_IDENTITY=-`). Never disable signing: the simulator Keychain needs the signed application identity.
- System frameworks only. Add a package only for a concrete unmet requirement, and include its cost in the cold-build measurement.
- Source must build with Xcode 26.6 (the CI image) as well as newer local Xcode. Guard anything newer than the iOS 26 SDK.

## Layout

- `Sources/Core`, `Sources/Networking`, `Sources/Servers`, `Sources/Auth`: JSON, transport, server origins, Keychain and session state.
- `Sources/Chat`: the chat API, the conversation controller and its model (`Model/`), ported from the reference client's protocol logic.
- `Sources/Company`, `Sources/Connections`, `Sources/Platform`: per-account stores, connection services and provider adapters.
- `Sources/UI`: SwiftUI views. Views hold no transport or protocol logic.
- `Sources/Fixtures`: Debug-only UI-protocol fixtures behind `UI_PROTOCOL_FIXTURES` and the `-AutoGPTUIProtocolFixtures YES` launch argument. Fixture behavior is never integration evidence.

## Rules

- Keep every accessibility identifier and visible string the Maestro journeys in `../e2e/` use. Add identifiers for new controls.
- Credentials: signed `set-auth-token` session only, Keychain partitioned by server and account, API tokens in memory only, no cookies, no redirects. Only the auth server's 401 ends a session. Treat every account or server change as a new generation.
- Never replay queued sends automatically. Never embed the desktop product in a web view; artifact web views stay isolated.
- Provider integrations (Google, APNs, App Store) stay behind adapters that report readiness; never show a control that can't complete.
- Format with `xcrun swift-format format --in-place --recursive Sources` and check with `xcrun swift-format lint --recursive Sources` before committing.
