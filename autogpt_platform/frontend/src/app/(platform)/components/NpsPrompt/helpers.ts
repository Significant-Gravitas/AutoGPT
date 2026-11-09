import { Key, storage } from "@/services/storage/local-storage";

const NPS_AFTER_MS = 14 * 24 * 60 * 60 * 1000;

export function isNpsDue(createdAt: string | undefined, now: number) {
  if (!createdAt) return false;
  const created = new Date(createdAt).getTime();
  return !Number.isNaN(created) && now - created >= NPS_AFTER_MS;
}

// Which account last had its one NPS ask on this browser, so most page loads
// make no call. The backend still enforces the once-ever limit.
export function hasNpsMarker(userID: string) {
  return storage.get(Key.NPS_DONE) === userID;
}

export function setNpsMarker(userID: string) {
  storage.set(Key.NPS_DONE, userID);
}
