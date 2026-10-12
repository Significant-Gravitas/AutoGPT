const KEY_PREFIX = "copilot-held-follow-ups:";

function storage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.sessionStorage;
  } catch {
    return null;
  }
}

function read(store: Storage, key: string): string[] {
  try {
    const parsed: unknown = JSON.parse(store.getItem(key) ?? "[]");
    return Array.isArray(parsed)
      ? parsed.filter((entry): entry is string => typeof entry === "string")
      : [];
  } catch {
    return [];
  }
}

// Best effort: a full or blocked storage must not stop the follow-up itself.
function write(store: Storage, key: string, texts: string[]) {
  try {
    if (texts.length === 0) store.removeItem(key);
    else store.setItem(key, JSON.stringify(texts));
  } catch (error) {
    console.warn("Could not persist held follow-ups", error);
  }
}

/**
 * A follow-up the page is holding until its own stream has settled (see
 * `useCopilotPage.onSend`) exists nowhere but in memory. These keep a copy in
 * sessionStorage for the chat it belongs to, so a reload or a switch to
 * another chat before it went out can hand it back to the composer.
 */
export function rememberHeldFollowUp(sessionId: string, text: string) {
  const store = storage();
  if (!store) return;
  const key = `${KEY_PREFIX}${sessionId}`;
  write(store, key, [...read(store, key), text]);
}

export function forgetHeldFollowUp(sessionId: string, text: string) {
  const store = storage();
  if (!store) return;
  const key = `${KEY_PREFIX}${sessionId}`;
  const texts = read(store, key);
  const index = texts.indexOf(text);
  if (index === -1) return;
  write(
    store,
    key,
    texts.filter((_, i) => i !== index),
  );
}

/** Returns and clears what was left behind for this chat. */
export function takeHeldFollowUps(sessionId: string): string[] {
  const store = storage();
  if (!store) return [];
  const key = `${KEY_PREFIX}${sessionId}`;
  const texts = read(store, key);
  write(store, key, []);
  return texts;
}
