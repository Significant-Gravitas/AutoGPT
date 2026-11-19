import type { ScheduleCreationRequest } from "@/app/api/__generated__/models/scheduleCreationRequest";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { createSafeStorage } from "@/services/storage/safe-storage";
import { v4 as uuidv4 } from "uuid";

// A schedule create carries a key that the server makes the schedule's id. A
// retry of the same create sends the same key, so when an earlier attempt was
// applied but its answer was lost, the server answers with that schedule
// instead of adding a second one that would run, and charge, again.
//
// The key is kept in sessionStorage under a hash of the create and the user,
// so a retry after a reload still sends it, a changed create gets a key of its
// own, and another account signed in to the same tab never inherits it.
// Where the browser blocks storage it lives in memory, so only until a reload.

type StorageKey = `schedule-create:${string}`;

const storage = createSafeStorage<StorageKey>("session");

// Refused for good: the key is taken by a different create (409), or the
// account may not create it (403). A retry must start over with a new key.
const FINAL_REFUSALS = new Set([403, 409]);

interface ScheduleCreate {
  graphId: string;
  data: ScheduleCreationRequest;
}

interface Args {
  userId: string | undefined;
  create: ScheduleCreate;
}

export function withScheduleCreateKey({
  userId,
  create,
}: Args): ScheduleCreate {
  const slot = slotFor({ userId, create });
  // uuidv4({}) rather than crypto.randomUUID, which is missing on plain HTTP.
  const key = storage.get(slot) ?? uuidv4({});
  storage.set(slot, key);
  return { ...create, data: { ...create.data, schedule_id: key } };
}

// Called with the create's answer. Success or a final refusal drops the key;
// anything else (a 503, a timeout, a lost connection) keeps it for the retry.
// Other 4xx answers keep it too: nothing was made, and the same create may
// still go through unchanged, e.g. once a slot is freed at the limit.
// A late answer leaves alone a key a newer attempt has taken since.
export function forgetScheduleCreateKey({
  userId,
  create,
  error,
}: Args & { error?: unknown }) {
  if (error !== undefined && !isFinalRefusal(error)) return;
  const slot = slotFor({ userId, create });
  if (storage.get(slot) === create.data.schedule_id) storage.clean(slot);
}

function isFinalRefusal(error: unknown) {
  return error instanceof ApiError && FINAL_REFUSALS.has(error.status);
}

function slotFor({ userId, create }: Args): StorageKey {
  const { schedule_id: _key, ...request } = create.data;
  const graphId = create.graphId;
  return `schedule-create:${hash(sortedJSON({ userId, graphId, request }))}`;
}

// Object keys sorted, so the same create hashes alike however its form state
// was assembled.
function sortedJSON(value: unknown): string {
  return JSON.stringify(value, (_, item: unknown) =>
    item && typeof item === "object" && !Array.isArray(item)
      ? Object.fromEntries(
          Object.keys(item)
            .sort()
            .map((key) => [key, (item as Record<string, unknown>)[key]]),
        )
      : item,
  );
}

// cyrb53, synchronous: crypto.subtle is missing on plain HTTP too. It keeps
// the create's inputs out of the storage key.
function hash(text: string): string {
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ code, 2654435761);
    h2 = Math.imul(h2 ^ code, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
}
