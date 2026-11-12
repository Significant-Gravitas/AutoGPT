import { createAsyncStoragePersister } from "@tanstack/query-async-storage-persister";
import type { Query, QueryClient } from "@tanstack/react-query";
import {
  persistQueryClient,
  type PersistedClient,
  type Persister,
} from "@tanstack/react-query-persist-client";
import Storage from "expo-sqlite/kv-store";

import { chatKeys } from "@/lib/api/chat";
import { expertKeys } from "@/lib/api/experts";

/** How long a saved copy may open the app before it is thrown away. */
export const PERSIST_MAX_AGE_MS = 24 * 60 * 60 * 1000;
// Bump when a saved shape changes, so an old copy is dropped instead of read.
const PERSIST_VERSION = "1";
const MAX_SAVED_CHATS = 8;
// A chat whose saved copy would pass this many characters is fetched instead.
const MAX_CHAT_CHARS = 400_000;

// The save waits here rather than in the storage persister, so a sign-out
// can cancel a save that is still waiting and the old account's copy never
// lands after it.
const SAVE_DELAY_MS = 1000;

const storagePersister = createAsyncStoragePersister({
  storage: Storage,
  key: "autogpt-query-cache",
  throttleTime: 0,
});

let waiting: PersistedClient | null = null;
let saveTimer: ReturnType<typeof setTimeout> | null = null;

function saveWaiting() {
  saveTimer = null;
  const client = waiting;
  waiting = null;
  if (client) void storagePersister.persistClient(trimClient(client));
}

/**
 * Saves the chat list, your experts and the last few chats you opened, so the
 * app and those chats open at once, even before the network answers.
 */
export const queryPersister: Persister = {
  persistClient(client) {
    waiting = client;
    saveTimer ??= setTimeout(saveWaiting, SAVE_DELAY_MS);
  },
  restoreClient: () => storagePersister.restoreClient(),
  async removeClient() {
    if (saveTimer !== null) clearTimeout(saveTimer);
    saveTimer = null;
    waiting = null;
    await storagePersister.removeClient();
  },
};

/**
 * Restores the signed-in user's saved copy and keeps saving it. The copy is
 * stamped with the user's id, so another account's copy is thrown away
 * instead of shown. Returns the function that stops saving.
 */
export function startCachePersistence(queryClient: QueryClient, userID: string) {
  const [unsubscribe] = persistQueryClient({
    queryClient,
    persister: queryPersister,
    maxAge: PERSIST_MAX_AGE_MS,
    buster: `${PERSIST_VERSION}:${userID}`,
    dehydrateOptions: { shouldDehydrateQuery: shouldPersistQuery },
  });
  return unsubscribe;
}

function shouldPersistQuery(query: Query) {
  if (query.state.status !== "success") return false;
  const [scope, kind] = query.queryKey;
  if (scope === expertKeys.list[0]) return query.queryKey.length === 1;
  return scope === chatKeys.sessions[0] && (kind === "sessions" || kind === "session");
}

// Only the most recent chats, and none so large they would slow every save.
function trimClient(client: PersistedClient): PersistedClient {
  const chats = client.clientState.queries
    .filter((query) => query.queryKey[1] === "session")
    .filter((query) => JSON.stringify(query.state.data).length <= MAX_CHAT_CHARS)
    .sort((a, b) => b.state.dataUpdatedAt - a.state.dataUpdatedAt)
    .slice(0, MAX_SAVED_CHATS);
  const kept = new Set(chats.map((query) => query.queryHash));
  return {
    ...client,
    clientState: {
      ...client.clientState,
      queries: client.clientState.queries.filter(
        (query) => query.queryKey[1] !== "session" || kept.has(query.queryHash),
      ),
    },
  };
}
