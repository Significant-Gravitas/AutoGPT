import { storage, Key } from "@/services/storage/local-storage";

export function wasShownToday(key: Key): boolean {
  return storage.get(key) === new Date().toDateString();
}

export function markShownToday(key: Key): void {
  storage.set(key, new Date().toDateString());
}
