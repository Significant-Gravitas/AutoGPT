# Mobile prototype progress

## October 7, 2026: expert chats and requests

[Expert workspace](ios-workspace-experts.png) and [Needs you](ios-workspace-needs-you.png) were captured inside the iOS app on the local iPhone 16 Pro / iOS 18.3 simulator. They render the actual React workspace with explicitly labeled Storybook/MSW test data, through a loopback proxy; no live account or model is connected. The provider setup is unconfigured.

Interactive verification reached the expert and request tabs, scrolled to the review controls, confirmed that decline requires a second tap, and observed the accepted fixture decision remove the review (count changed from two to one). Questions link back to their original conversation. The screenshots and interaction prove the mobile UI with fixtures, not production chat or APNs/FCM delivery. See [push setup](../PUSH_SETUP.md).

Run `STORYBOOK_MOBILE_FIXTURE=true pnpm exec storybook dev -p 6006 --ci --no-open` from the frontend to reproduce the `Mobile/Workspace` stories. The existing native integration fixture separately exposes **Check native push** for a real WebView-to-native status round trip without requesting notification permission.

## October 7, 2026: native design-system screens

These unmodified CI captures show the shared AutoGPT logo, Poppins/Geist typography, 46-point minimum actions, and the current white bordered secondary-button style.

| Platform | Capture revision | Runtime                                                 | Screens                                                                                                                            |
| -------- | ---------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| iOS      | `12894670771`    | iPhone 16 Pro, iOS 18.5 simulator                       | [Sign-in](ios-design-sign-in.png), [server settings](ios-design-server-settings.png)                                               |
| Android  | `c3d9e032ffa`    | Pixel 7 Pro profile, API 36 Google APIs x86_64 emulator | [Sign-in](android-design-sign-in.png), [server settings](android-design-server-settings.png), [recovery](android-design-error.png) |

The screens use deterministic debug previews without a real signed-in account. Android's settings include a debug-only explanation of local HTTP addresses. These captures verify native layout; they do not prove live chat, provider sign-in, or the target physical devices.

The [iOS run](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/37621834368) completed three UI tests. Visual review found that its settings sheet did not inherit the test's large-text override; the test harness now applies that override to the window and asserts the heading is enlarged. The screenshots retained here are the normal-text portrait captures.

The [Android run](https://github.com/Significant-Gravitas/AutoGPT/actions/runs/37622966997) passed all five framework probes: WebView capability, cookie-deletion completion, fixture authentication and cancellation, native-download origin checks, and actual provider file saving/cancellation. It used WebView `133.0.6943.137`. Raw probe results and the preview captures are available in the run's `android-runtime-results` artifact.

## September 7, 2026: initial runtime evidence

These are unmodified screenshots captured from the running iOS app on the installed **iPhone 16 Pro simulator, iOS 18.3**, built against the iOS 26.5 SDK. They are not iPhone 17 Pro runtime evidence.

- `ios-sign-in.png`: native sign-in screen reached after the real `platform.agpt.co/copilot` login redirect. No user account is signed in.
- `ios-fixture-session.png`: successful system-browser return with both HttpOnly session and cache cookies transferred. The page explicitly identifies itself as a local integration fixture; it is not a live AutoGPT conversation.

Further fixture checks on the same simulator:

- `ios-fixture-streaming.png`: five server-sent chunks reached the native WebView incrementally, about 600 ms apart. This is transport test text; no model response is involved.
- `ios-fixture-stream-interrupted.png`: a deliberate disconnect after two chunks retains partial text and offers retry.
- `ios-fixture-keyboard.png`: text entered using the simulator's onscreen keyboard stays visible in portrait. The subsequent native fix dismisses the keyboard on rotation while preserving the draft; tapping the field again in landscape was verified to reveal it above the keyboard.

- `ios-fixture-file-roundtrip.png`: a generated Markdown file saved through the native share sheet into Files is selected again through the web attachment picker, retaining its filename and 69-byte size.
