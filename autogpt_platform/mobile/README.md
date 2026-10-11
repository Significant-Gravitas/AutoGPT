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

`.github/workflows/platform-native-ios-ci.yml` builds an arm64 simulator app on the standard free `macos-26` runner with Xcode 26.6 and runs a Maestro launch journey. Successful runs retain an installable `AutoGPT-simulator.zip`, build/resource logs, JUnit report and screenshots. This first launch gate does not certify real chat, all card actions, provider integrations or the complete native interaction/build feasibility gate.

The functional-test build uses the Release runtime and bundled production JavaScript, with native optimization and debug symbol generation disabled. `bash scripts/build-simulator.sh` reproduces those compiler settings locally. Swift compiles as a whole module to avoid repeated per-file frontend work. All configured native modules remain included. This artifact is for simulator behavior checks; optimized release performance and symbolicated native crashes require the ordinary Release build separately.

Three cold runner measurements hit the ten-minute limit before native tests:

| Commit / run                                                                                 | Setup | Build before cancellation | Whole job, including cleanup | Native tests |
| -------------------------------------------------------------------------------------------- | ----- | ------------------------- | ---------------------------- | ------------ |
| [`54f2a47`](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/38163767681)        | 3m29s | 6m38s                     | 10m17s                       | Skipped      |
| [`273ffa5`](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/38165924664)        | 2m46s | 7m27s                     | 11m07s                       | Skipped      |
| [`a80ca6f`](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/38167323587), Intel | 3m32s | 6m41s                     | 12m32s                       | Skipped      |

The second run overlapped simulator boot with compilation for 2m53s and progressed through fewer compilation operations. Resource contention is a hypothesis; those runs did not capture memory or CPU samples. [GitHub's standard runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) give the ARM runner 3 CPUs/7 GB and the Intel runner 4 CPUs/14 GB; both are free for public repositories. All bundled native frameworks include x86_64 simulator slices.

The Intel trial stalled in build script phases before any source compilation. Its one-minute host load peaked at 626.70, with hundreds of runnable processes after simulator startup. The samples do not identify the cause. The simulator selector also chose iOS 26.4 from JSON order although Xcode's SDK was 26.5. Runtime selection now requires an available iPhone matching the selected SDK and fails if none exists.

The next controlled measurement returns to standard ARM and compiles before booting the simulator, with CPU-focused resource samples that disable expensive shared-framework accounting. This separates source-build cost from simulator startup. The intended native modules and ten-minute limit stay in place, and launch tests must still execute and pass. A cold pass and a warm pass including setup, tests, artifact upload and cache saves remain required. If compilation itself exhausts the budget under normal runner load, evaluate the SwiftUI fallback before expanding the card surface. Abnormal runner behavior needs to be diagnosed separately; local or warm-only results are not acceptance.

The parent server lane has separate production-auth and PostgreSQL/Redis journeys; see [server validation](../../docs/engineering/native-mobile/server-validation.md). Real Google login, APNs and App Store purchase settlement need their own provider setup and evidence.

The [native E2E guide](e2e/README.md) also covers the local production-server sign-in, streaming and transcript-restoration journey.
