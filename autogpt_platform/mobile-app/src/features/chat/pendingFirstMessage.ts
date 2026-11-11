// The message typed on Home, held until the new chat's screen mounts and
// sends it. Taken once, so coming back to the screen never sends it again.
export interface FirstMessage {
  text: string;
  files: { id: string; name: string }[];
}

const pending = new Map<string, FirstMessage>();

export function setPendingFirstMessage(sessionID: string, message: FirstMessage) {
  pending.set(sessionID, message);
}

export function takePendingFirstMessage(sessionID: string) {
  const message = pending.get(sessionID);
  pending.delete(sessionID);
  return message;
}
