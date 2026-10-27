// A denied permission is the user's or a policy's answer; asking again cannot
// change it. Anything else (no microphone, one held by another app) can be
// fixed on the device, so callers tell the two apart.
export function isMicrophonePermissionDenied(error: unknown) {
  return error instanceof DOMException && error.name === "NotAllowedError";
}

export function microphoneErrorName(error: unknown) {
  return error instanceof Error ? error.name : "unknown";
}
