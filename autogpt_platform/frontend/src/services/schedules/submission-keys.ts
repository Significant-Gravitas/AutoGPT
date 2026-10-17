// Idempotency keys for schedule creates. Retrying the same payload reuses its
// key, so a create the scheduler applied without confirming isn't doubled; a
// changed payload is a new submission and gets a new key.
//
// The pending key lives in sessionStorage, keyed by a hash of the payload
// (which includes the agent), so a remount or reload of the tab after an
// unconfirmed create still resends it. It is dropped once a create succeeds.
const STORAGE_PREFIX = "schedule-create-key:";

const memoryFallback = new Map<string, string>();

export function createSubmissionKeys() {
  let lastStorageKey: string | null = null;

  return {
    keyFor(payload: unknown): string {
      const storageKey = STORAGE_PREFIX + hashString(JSON.stringify(payload));
      lastStorageKey = storageKey;
      const existing = readKey(storageKey);
      if (existing) return existing;
      const key = crypto.randomUUID();
      writeKey(storageKey, key);
      return key;
    },
    reset() {
      if (lastStorageKey) removeKey(lastStorageKey);
      lastStorageKey = null;
    },
  };
}

function readKey(storageKey: string): string | null {
  try {
    return window.sessionStorage.getItem(storageKey);
  } catch {
    return memoryFallback.get(storageKey) ?? null;
  }
}

function writeKey(storageKey: string, key: string) {
  try {
    window.sessionStorage.setItem(storageKey, key);
  } catch {
    memoryFallback.set(storageKey, key);
  }
}

function removeKey(storageKey: string) {
  try {
    window.sessionStorage.removeItem(storageKey);
  } catch {
    memoryFallback.delete(storageKey);
  }
}

// cyrb53: a fast 53-bit string hash. Keeps form values (which can be
// sensitive) out of the storage key itself.
function hashString(value: string): string {
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < value.length; i++) {
    const ch = value.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
}
