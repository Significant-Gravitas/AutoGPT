# Mobile prototype progress

These are unmodified screenshots captured from the running iOS app on the installed **iPhone 16 Pro simulator, iOS 18.3**, built against the iOS 26.5 SDK. They are not iPhone 17 Pro runtime evidence.

- `ios-sign-in.png`: native sign-in screen reached after the real `platform.agpt.co/copilot` login redirect. No user account is signed in.
- `ios-fixture-session.png`: successful system-browser return with both HttpOnly session and cache cookies transferred. The page explicitly identifies itself as a local integration fixture; it is not a live AutoGPT conversation.

Further fixture checks on the same simulator:

- `ios-fixture-streaming.png`: five server-sent chunks reached the native WebView incrementally, about 600 ms apart. This is transport test text; no model response is involved.
- `ios-fixture-stream-interrupted.png`: a deliberate disconnect after two chunks retains partial text and offers retry.
- `ios-fixture-keyboard.png`: text entered using the simulator's onscreen keyboard stays visible in portrait. The subsequent native fix dismisses the keyboard on rotation while preserving the draft; tapping the field again in landscape was verified to reveal it above the keyboard.

- `ios-fixture-file-roundtrip.png`: a generated Markdown file saved through the native share sheet into Files is selected again through the web attachment picker, retaining its filename and 69-byte size.
