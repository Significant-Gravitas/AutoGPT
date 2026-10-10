# Android runtime evidence

Captured September 7, 2026, using Android Emulator 37.1.11, an API 36 Google APIs arm64 revision 7 image, a Pixel 9 Pro device profile, and Google WebView 133.0.6943.137. This is separate from the intended Pixel 11 Pro physical-device target.

The final tested debug APK SHA256 is `7ec35067facfb0ed84c47b266e235f696a1bc32291724e5390c1e7c0d04acc15`. The screenshot was captured from commit `dc4b0ddd93`. The final tested app adds the narrow orphan-staging cleanup, and its instrumentation adds the provider save/cancellation check committed with the updated evidence.

[Native Sign in screen](android-api36-sign-in.png) shows the installed app after the real hosted `https://platform.agpt.co/copilot` redirected to login. It does not show an authenticated production chat. The larger primary and secondary controls retain native button behavior.

[Raw instrumentation result](android-api36-runtime.txt) records five passing checks on the actual Android WebView: required platform features; full-cookie deletion completion ordering; local-fixture native authentication exchange, cookie-family installation, old-cookie removal, and cancellation preservation; native download messaging with trusted top-frame cancellation, malformed input rejection, iframe denial, and foreign-origin bridge absence; and saving 98,321 exact bytes through three acknowledged chunks into a test-owned pending MediaStore row, preserving those bytes when a later transfer is canceled, and deleting only that owned row afterward. The local server was verified as the labelled disposable fixture at `http://127.0.0.1:8765`, connected with `adb reverse`.

The instrumentation created no browser windows or OS pickers and used no real account. Picker selection/cancellation was simulated at the native picker callback. The real ContentResolver and ParcelFileDescriptor copy path was exercised without granting permissions. The final app-private native staging directory was verified empty. Browser return UI, real save/upload dialogs, keyboard interactions, microphone capture, and physical-device behavior remain unverified on Android.

The debug/release builds, test APK, 29 JVM unit tests, and debug/release lint passed. Lint reported no errors; seven dependency/Gradle update notices remain. The additional filesystem tests verify that startup cleanup removes only matching stale files and preserves unrelated, nested, and current-process files. See the Android README for reproducible instrumentation commands and the explicit disposable-device guard.
