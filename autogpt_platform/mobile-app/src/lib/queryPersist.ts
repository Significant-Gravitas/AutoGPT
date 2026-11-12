import { createAsyncStoragePersister } from "@tanstack/query-async-storage-persister";
import type { Query } from "@tanstack/react-query";
import type { PersistedClient, Persister } from "@tanstack/react-query-persist-client";
import Storage from "expo-sqlite/kv-store";

import { chatKeys } from "@/lib/api/chat";
import { expertKeys } from "@/lib/api/experts";

/** How long a saved copy may open the app before it is thrown away. */
export const PERSIST_MAX_AGE_MS = 24 * 60 * 60 * 1000;
// Bump when a saved shape changes, so an old copy is dropped instead of read.
export const PERSIST_BUSTER = "1";
const MAX_SAVED_CHATS = 8;
// A chat whose saved copy would pass this many characters is fetched instead.
const MAX_CHAT_CHARS = 400_000;

const storagePersister = createAsyncStoragePersister({
  storage: Storage,
  key: "autogpt-query-cache",
  throttleTime: 1000,
});

/**
 * Saves the chat list, your experts and the last few chats you opened, so the
 * app and those chats open at once, even before the network answers.
 */
export const queryPersister: Persister = {
  persistClient: (client) => storagePersister.persistClient(trimClient(client)),
  restoreClient: () => storagePersister.restoreClient(),
  removeClient: () => storagePersister.removeClient(),
};

export function shouldPersistQuery(query: Query) {
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
