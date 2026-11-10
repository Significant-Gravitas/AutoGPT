// The message typed on Home, held until the new chat's screen mounts and
// sends it. Taken once, so coming back to the screen never sends it again.
const pending = new Map<string, string>();

export function setPendingFirstMessage(sessionID: string, text: string) {
  pending.set(sessionID, text);
}

export function takePendingFirstMessage(sessionID: string) {
  const text = pending.get(sessionID);
  pending.delete(sessionID);
  return text;
}
