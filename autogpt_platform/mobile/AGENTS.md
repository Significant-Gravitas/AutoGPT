# Native AutoGPT experiment

Read the root and platform contribution guides and `docs/engineering/native-mobile/implementation-plan.md`. This is the iOS-first app layer, stacked on the native server changes. Android follows the working iOS experience.

## Product and architecture

- Native Chats and Company destinations. No Build, Library, or account profile editing.
- Use Expo Router in `src/app/`; keep transport, state and platform adapters outside route components.
- Use native controls, navigation, system typography, light/dark appearance, Dynamic Type and accessible touch targets. Frontend DOM/Tailwind components are reference material, not dependencies of this app.
- All tool cards and actions remain in scope. A generic details view is a compatibility fallback, not proof that a card's actions are implemented.
- Never put the desktop product in a WebView. Artifact WebViews must be isolated from app credentials and privileged native bridges.
- Native credentials live in SecureStore, partitioned by server/account. Credentials must not enter logs, URLs, screenshots or test artifacts.
- Do not automatically replay queued sends until the server's durable cancellation contract is complete.

## Toolchain

Use Node 24 and the pinned pnpm version. Expo APIs change between SDKs: read the version in `package.json`, then the matching `https://docs.expo.dev/versions/v<major>.0.0/` documentation before introducing or changing an Expo API. Use `https://docs.expo.dev/llms.txt` for other official guides.

```sh
pnpm install --frozen-lockfile
pnpm expo install <package>
pnpm format
pnpm lint
pnpm types
pnpm prebuild:ios
pnpm ios
```

Use `expo install` for SDK-compatible native modules. Keep `ios/` and `android/` generated and ignored; change native configuration through `app.config.ts` and config plugins. No EAS service, OTA update, signing or store submission is required for this experiment.

## Validation

- Prioritize native E2E journeys. Mock external model/provider boundaries, never production auth, persistence or card insertion, when calling a journey integration coverage.
- Build with the intended native-module set. Measure cold and warm builds on a standard free macOS runner; the workflow critical path must stay below ten minutes. A timeout is not a successful measurement.
- Local disk and RAM are constrained. Avoid duplicate backend/frontend dependencies and concurrent native builds. Preserve other worktrees, running services and simulators.
- Use the shared T3 Device panel for local verification. Save the same real screenshot evidence in the Linear spike and user conversation.
- Run whole-app formatting, lint and types before committing. Retain test reports and screenshots without credentials.
