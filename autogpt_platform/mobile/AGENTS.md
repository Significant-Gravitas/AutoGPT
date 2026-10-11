# Native AutoGPT experiment

Read the root and platform contribution guides and `docs/engineering/native-mobile/implementation-plan.md`. This is the iOS-first app layer, stacked on the native server changes. The current direction is the planned SwiftUI fallback under `apple/` after the React Native cold-build gate failed. The existing React Native source remains reference material during the spike. Android follows as a separate client sharing the server contract and fixtures.

## Product and architecture

- Native Chats and Company destinations. No Build, Library, or account profile editing.
- Use SwiftUI/UIKit in `apple/`; keep transport, state and platform adapters outside views. Preserve the existing Maestro accessibility identifiers during the port.
- Use native controls, navigation, system typography, light/dark appearance, Dynamic Type and accessible touch targets. Frontend DOM/Tailwind components are reference material, not dependencies of this app.
- All tool cards and actions remain in scope. A generic details view is a compatibility fallback, not proof that a card's actions are implemented.
- Never put the desktop product in a WebView. Artifact WebViews must be isolated from app credentials and privileged native bridges.
- Native credentials live in Keychain, partitioned by server/account. Credentials must not enter logs, URLs, screenshots or test artifacts.
- Do not automatically replay queued sends until the server's durable cancellation contract is complete.

## Toolchain

Use Xcode 26.6-compatible Swift and system frameworks for the new Apple client. Keep a reproducible Xcode project and its project-generation specification together; no Apple signing account is needed for an ad hoc signed simulator build. Avoid introducing package/build-time code generators until their cost is included in the cold measurement. Format all Swift sources with the selected Swift formatter and build before committing.

The reference React Native client uses Node 24 and the pinned pnpm version. For changes to that client, read the version in `package.json`, then matching Expo documentation before changing an Expo API.

```sh
pnpm install --frozen-lockfile
pnpm expo install <package>
pnpm format
pnpm lint
pnpm types
pnpm prebuild:ios
pnpm ios
```

Keep the reference client's generated `ios/` and `android/` directories ignored. The new SwiftUI source and project live in `apple/`, not the generated `ios/` directory. No EAS service, OTA update, store signing or submission is required for this experiment.

## Validation

- Prioritize native E2E journeys. Mock external model/provider boundaries, never production auth, persistence or card insertion, when calling a journey integration coverage.
- Build with all intended frameworks and packages. Measure cold and warm builds on a standard free macOS runner; the workflow critical path must stay below ten minutes. A timeout is not a successful measurement. Do not overlap simulator first boot with compilation on those runners.
- Local disk and RAM are constrained. Avoid duplicate backend/frontend dependencies and concurrent native builds. Preserve other worktrees, running services and simulators.
- Use the shared T3 Device panel for local verification. Save the same real screenshot evidence in the Linear spike and user conversation.
- Run whole-app formatting and relevant compile/lint checks before committing. The reference TypeScript client still uses its format/lint/types commands. Retain test reports and screenshots without credentials.
