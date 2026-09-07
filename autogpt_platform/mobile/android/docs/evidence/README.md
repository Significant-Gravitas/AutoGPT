# Android runtime evidence

Captured September 7, 2026, using Android Emulator 37.1.11, an API 36 Google APIs arm64 revision 7 image, a Pixel 9 Pro device profile, and Google WebView 133.0.6943.137. This is separate from the intended Pixel 11 Pro physical-device target.

The tested debug APK SHA256 is `e29eeb04336ae9e1ddcb08753c95cf0929e89308ad3fe07eedb05517a324225c`. The source was based on `8ea536e6a7` plus the Android runtime-probe and native status-action changes committed with this evidence.

[Native Sign in screen](android-api36-sign-in.png) shows the installed app after the real hosted `https://platform.agpt.co/copilot` redirected to login. It does not show an authenticated production chat. The larger primary and secondary controls retain native button behavior.

[Raw instrumentation result](android-api36-runtime.txt) records four passing checks on the actual Android WebView: required platform features; full-cookie deletion completion ordering; local-fixture native authentication exchange, cookie-family installation, old-cookie removal, and cancellation preservation; and native download messaging with trusted top-frame cancellation, malformed input rejection, iframe denial, and foreign-origin bridge absence. The local server was verified as the labelled disposable fixture at `http://127.0.0.1:8765`, connected with `adb reverse`.

The instrumentation created no browser windows or OS pickers and used no real account. Its picker cancellation was simulated at the native picker callback. Browser return UI, real save/upload dialogs, keyboard interactions, microphone capture, and physical-device behavior remain unverified on Android.

The debug/release builds, test APK, 27 JVM unit tests, and debug/release lint passed. Lint reported no errors; seven dependency/Gradle update notices remain. See the Android README for reproducible instrumentation commands and the explicit disposable-device guard.
