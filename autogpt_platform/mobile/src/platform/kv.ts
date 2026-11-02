import Storage from "expo-sqlite/kv-store";

/**
 * Non-secret app state lives in the app container, which iOS deletes on
 * uninstall (unlike the Keychain). Credentials never go here.
 */
export async function readJSON<T>(
  key: string,
  isValid: (value: unknown) => value is T,
): Promise<T | null> {
  const raw = await Storage.getItemAsync(key);
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    return isValid(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export async function writeJSON(key: string, value: unknown): Promise<void> {
  await Storage.setItemAsync(key, JSON.stringify(value));
}

export async function removeKey(key: string): Promise<void> {
  await Storage.removeItemAsync(key);
}

export async function readFlag(key: string): Promise<boolean> {
  return (await Storage.getItemAsync(key)) === "1";
}

export async function writeFlag(key: string): Promise<void> {
  await Storage.setItemAsync(key, "1");
}
