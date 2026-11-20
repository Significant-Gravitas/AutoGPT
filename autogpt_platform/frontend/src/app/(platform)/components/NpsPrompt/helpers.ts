import { Key, storage } from "@/services/storage/local-storage";

const DAY_MS = 24 * 60 * 60 * 1000;
const NPS_AFTER_MS = 14 * DAY_MS;

export function isNpsDue(createdAt: string | undefined, now: number) {
  if (!createdAt) return false;
  const created = new Date(createdAt).getTime();
  return !Number.isNaN(created) && now - created >= NPS_AFTER_MS;
}

// Which account last checked on this browser: just its id once it was asked,
// or `id|time` when the backend said not yet (it also wants a paying account
// that got something done this week), so most page loads make no call. The
// backend still enforces every limit.
export function shouldCheckNps(userID: string, now: number) {
  const [owner, nextCheck] = (storage.get(Key.NPS_CHECK) ?? "").split("|");
  if (owner !== userID) return true;
  return nextCheck !== undefined && now >= Number(nextCheck);
}

export function markNpsAsked(userID: string) {
  storage.set(Key.NPS_CHECK, userID);
}

export function markNpsNotYet(userID: string, now: number) {
  storage.set(Key.NPS_CHECK, `${userID}|${now + DAY_MS}`);
}
