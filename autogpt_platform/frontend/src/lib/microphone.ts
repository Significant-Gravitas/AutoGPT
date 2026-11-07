// A NotAllowedError is a permission answer, but not always a final one: Chrome
// returns it when the prompt was only closed, and a blocked site can be allowed
// from the address bar, so asking again can work. Anything else (no microphone,
// one held by another app) has to be fixed on the device, so callers tell the
// two apart.
export function isMicrophonePermissionDenied(error: unknown) {
  return error instanceof DOMException && error.name === "NotAllowedError";
}

export function microphoneErrorName(error: unknown) {
  return error instanceof Error ? error.name : "unknown";
}
