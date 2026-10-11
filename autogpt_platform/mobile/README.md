# AutoGPT native experiment

iOS-first native app for Otto, your AI company and their conversations. This is an experiment, not a released mobile client. The full scope, card inventory and acceptance gates live in the [implementation plan](../../docs/engineering/native-mobile/implementation-plan.md).

The app follows the [platform license](../LICENSE.md). The [Expo template notice](licenses/expo-template-LICENSE) preserves attribution for the generated starting template.

## Local development

Use Node 24.18 or newer within Node 24, pnpm 10.20, Xcode 26.4 or newer and CocoaPods. The app targets iOS 17 and newer; the current shared simulator runs iOS 27. Android is a follow-up and is not needed for iOS development.

```sh
pnpm install --frozen-lockfile
pnpm ios
```

The native projects are generated from `app.config.ts`; do not edit or commit `ios/` and `android/`. `pnpm prebuild:ios` updates the iOS project without clearing restored CocoaPods or installing them. Scene support is enabled through `expo-build-properties` for iOS 27. Development does not require an Expo cloud account, OTA updates or App Store signing.

Simulator builds use Xcode's ad hoc signing (`CODE_SIGN_IDENTITY=-`). Keep signing enabled: disabling it removes the simulator's application identity and makes Keychain access fail. No Apple account, signing certificate or provisioning profile is required.

The app has its own experimental bundle identifier, `com.agpt.mobile.experimental`. It can coexist with the retired web experiment. Never use a production account in deterministic test fixtures.

## Checks and build evidence

```sh
pnpm format
pnpm lint
pnpm types
```

`.github/workflows/platform-native-ios-ci.yml` builds an arm64 simulator app on the standard free `macos-26` runner with Xcode 26.6 and runs a Maestro launch journey. The artifact contains the installable `AutoGPT-simulator.zip`, build logs, JUnit report and screenshots. This first launch gate does not certify real chat, all card actions, provider integrations or the complete native interaction/build feasibility gate.

The functional-test build uses the Release runtime and bundled production JavaScript, with native optimization and debug symbol generation disabled. `bash scripts/build-simulator.sh` reproduces those compiler settings locally. Swift compiles as a whole module to avoid repeated per-file frontend work. All configured native modules remain included. This artifact is for simulator behavior checks; optimized release performance and symbolicated native crashes require the ordinary Release build separately.

The [first cold iOS run](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/38163767681) hit the ten-minute limit during compilation: dependency setup took 3m29s and compilation was still running after another 6m38s. No native tests ran. The next measurement uses the functional-test compiler settings and overlaps Maestro/simulator preparation with compilation; a passing cold and warm run is still required. The parent server lane has separate production-auth and PostgreSQL/Redis journeys; see [server validation](../../docs/engineering/native-mobile/server-validation.md). Real Google login, APNs and App Store purchase settlement need their own provider setup and evidence.

The [native E2E guide](e2e/README.md) also covers the local production-server sign-in, streaming and transcript-restoration journey.
