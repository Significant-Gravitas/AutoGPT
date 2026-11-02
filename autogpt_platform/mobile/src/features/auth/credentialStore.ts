import * as SecureStore from "expo-secure-store";

import { readFlag, writeFlag } from "../../platform/kv";

/**
 * A signed Better Auth session credential, bound to the origins it was issued
 * for. It is only ever sent to `authOrigin`. Keychain items are partitioned
 * by server and account, readable only while the device is unlocked, and are
 * not migrated to other devices.
 */
export interface StoredCredential {
  version: 1;
  sessionToken: string;
  authOrigin: string;
  apiOrigin: string;
  userID: string;
  email: string;
  name: string | null;
  savedAt: string;
}

const INDEX_KEY = "autogpt.credentials.index.v1";
const INSTALL_MARKER = "installation.v1";
const OPTIONS: SecureStore.SecureStoreOptions = {
  keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
};

export function credentialKey(serverKey: string, accountKey: string): string {
  return `autogpt.credential.v1.${serverKey}.${accountKey}`;
}

export async function readCredential(
  key: string,
): Promise<StoredCredential | null> {
  const raw = await SecureStore.getItemAsync(key, OPTIONS);
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    return isStoredCredential(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export async function writeCredential(
  key: string,
  credential: StoredCredential,
): Promise<void> {
  await SecureStore.setItemAsync(key, JSON.stringify(credential), OPTIONS);
  const index = await readIndex();
  if (!index.includes(key)) await writeIndex([...index, key]);
}

export async function deleteCredential(key: string): Promise<void> {
  await SecureStore.deleteItemAsync(key, OPTIONS);
  const index = await readIndex();
  if (index.includes(key)) await writeIndex(index.filter((k) => k !== key));
}

/**
 * iOS keeps Keychain items after the app is deleted. The install marker lives
 * in the app container, so its absence means a fresh install: any session a
 * previous install left behind is removed before anything can use it.
 */
export async function wipeCredentialsLeftByPreviousInstall(): Promise<void> {
  if (await readFlag(INSTALL_MARKER)) return;
  const index = await readIndex();
  for (const key of index) {
    await SecureStore.deleteItemAsync(key, OPTIONS).catch(() => {});
  }
  await SecureStore.deleteItemAsync(INDEX_KEY, OPTIONS).catch(() => {});
  await writeFlag(INSTALL_MARKER);
}

async function readIndex(): Promise<string[]> {
  const raw = await SecureStore.getItemAsync(INDEX_KEY, OPTIONS);
  if (!raw) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.filter((k): k is string => typeof k === "string")
      : [];
  } catch {
    return [];
  }
}

async function writeIndex(keys: string[]): Promise<void> {
  await SecureStore.setItemAsync(INDEX_KEY, JSON.stringify(keys), OPTIONS);
}

function isStoredCredential(value: unknown): value is StoredCredential {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    v.version === 1 &&
    typeof v.sessionToken === "string" &&
    v.sessionToken.includes(".") &&
    typeof v.authOrigin === "string" &&
    typeof v.apiOrigin === "string" &&
    typeof v.userID === "string" &&
    typeof v.email === "string" &&
    (v.name === null || typeof v.name === "string") &&
    typeof v.savedAt === "string"
  );
}
