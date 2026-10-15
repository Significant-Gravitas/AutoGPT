# AutoGPT native experiment

iOS-first native app for Otto, your AI company and their conversations. This is an experiment, not a released mobile client. The full scope, card inventory and acceptance gates live in the [implementation plan](../../docs/engineering/native-mobile/implementation-plan.md).

The iOS client uses SwiftUI under [`apple/`](apple/README.md), following the planned fallback after four React Native cold builds missed the ten-minute gate. The `src/` React Native client remains reference material. Android will share the server contract and fixtures but needs its own UI.

The app follows the [platform license](../LICENSE.md). The [Expo template notice](licenses/expo-template-LICENSE) preserves attribution for the generated starting template.

## Local development

Use Xcode 26.6 or newer. The app targets iOS 17 and newer; the shared simulator currently runs iOS 27. From this folder:

```sh
bash scripts/build-simulator.sh
```

This builds optimized Release and fixture-enabled Debug apps under `apple/.build/SimulatorChecks/Build/Products/`. Open `apple/AutoGPT.xcodeproj` to develop in Xcode. The project is committed alongside its XcodeGen specification; only changes to that specification or the file list need `xcodegen generate` inside `apple/`. System frameworks only: no package install or CocoaPods step is needed for this client.

The reference React Native source still uses Node 24.18, pnpm 10.20 and `pnpm ios`; its generated `ios/` and `android/` folders stay ignored. No Expo cloud account, OTA updates or App Store signing is used.

Simulator builds use Xcode's ad hoc signing (`CODE_SIGN_IDENTITY=-`). Keep signing enabled: disabling it removes the simulator's application identity and makes Keychain access fail. No Apple account, signing certificate or provisioning profile is required.

The app has its own experimental bundle identifier, `com.agpt.mobile.experimental`. It can coexist with the retired web experiment. Never use a production account in deterministic test fixtures.

## Checks and build evidence

```sh
pnpm format
pnpm lint
pnpm types
```

`.github/workflows/platform-native-ios-ci.yml` compiles both SwiftUI configurations on the standard free `macos-26` ARM runner with Xcode 26.6. Builds use the host architecture; set `NATIVE_SIMULATOR_ARCHS='arm64 x86_64'` only when a universal simulator artifact is needed. The workflow disables host Spotlight indexing on its disposable runner, runs the launch journey against Release, then native UI-protocol interaction fixtures against Debug. No dependency or DerivedData caches are restored. The entire job, including setup, compilation, journeys and artifact upload, keeps a ten-minute timeout. The independent Linux lane checks the reference TypeScript client.

Successful runs retain the Release `AutoGPT-simulator.zip`, compiler/resource logs, JUnit reports and screenshots. Debug fixture routes are excluded from Release. Local compilation and journeys do not establish the free-runner timing gate; its first cold pass and subsequent measurement remain pending. The interaction fixtures do not certify production-server chat or provider integrations.

The [first SwiftUI ARM run](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/38559995734/job/115838410189) compiled Release in 49.47s and Debug in 21.18s under Xcode 26.6. Simulator preparation took 6m03s, including 5m39s in boot; most boot time was spent in Apple's data migration. The job exceeded the ten-minute limit during the Release launch journey and skipped the Debug fixtures. Including cancellation and cleanup, the failed job lasted 12m37s. Its final process snapshot had 315 runnable processes on three CPUs; this suggests resource contention but does not establish its cause.

The [SwiftUI Intel trial](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/38710418802/job/116340129610) built universal Release in 256.63s and Debug in 88.31s. Boot took 4m59s and the job exceeded ten minutes during simulator preparation; both journeys were skipped. Boot samples reached a one-minute load of 548.68, while swap snapshots and counters remained zero. Host Spotlight workers competed with compilation, and simulator system processes dominated boot. This does not identify a particular simulator defect. The next run returns to native ARM builds, disables host indexing, and retains bounded simulator system error logs alongside resource samples. Crash reporting and the test assertions remain enabled. The full cold/warm gate remains open until successful measured runs.

### Retired React Native build measurements

Four cold runner measurements hit the ten-minute limit before native tests:

| Commit / run                                                                                      | Setup | Build before cancellation | Whole job, including cleanup | Native tests |
| ------------------------------------------------------------------------------------------------- | ----- | ------------------------- | ---------------------------- | ------------ |
| [`54f2a47`](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/38163767681)             | 3m29s | 6m38s                     | 10m17s                       | Skipped      |
| [`273ffa5`](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/38165924664)             | 2m46s | 7m27s                     | 11m07s                       | Skipped      |
| [`a80ca6f`](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/38167323587), Intel      | 3m32s | 6m41s                     | 12m32s                       | Skipped      |
| [`5b5e85c`](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/38169435743), ARM serial | 2m09s | 8m04s                     | 10m29s                       | Skipped      |

The second run overlapped simulator boot with compilation for 2m53s and progressed through fewer compilation operations. Resource contention is a hypothesis; those runs did not capture memory or CPU samples. [GitHub's standard runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) give the ARM runner 3 CPUs/7 GB and the Intel runner 4 CPUs/14 GB; both are free for public repositories. All bundled native frameworks include x86_64 simulator slices.

The Intel trial stalled in build script phases before any source compilation. Its one-minute host load peaked at 626.70, with hundreds of runnable processes after simulator startup. The samples do not identify the cause. The simulator selector also chose iOS 26.4 from JSON order although Xcode's SDK was 26.5. Runtime selection now requires an available iPhone matching the selected SDK and fails if none exists.

The final controlled run returned to ARM and compiled before any simulator boot. It began 304 C/C++ compilation operations and four Swift compilations but exhausted the budget before linking. Neither tool cache was restored. That result, alongside Opus's source-build inventory, triggered the planned SwiftUI fallback. These are incomplete build measurements, not completed RN build-time benchmarks.

The parent server lane has separate production-auth and PostgreSQL/Redis journeys; see [server validation](../../docs/engineering/native-mobile/server-validation.md). Real Google login, APNs and App Store purchase settlement need their own provider setup and evidence.

The [native E2E guide](e2e/README.md) also covers the local production-server sign-in, streaming and transcript-restoration journey.
