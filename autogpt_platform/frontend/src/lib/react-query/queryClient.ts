"use client";

import { isServer, QueryClient } from "@tanstack/react-query";

function makeQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60 * 1000, // Added this because if staleTime is 0 (default), the data fetched on the server becomes stale immediately on the client, and it refetches again.

        // Highlighting some important defaults to avoid confusion for new devs
        // Queries are stale by default → triggers background refetch
        // Refetch triggers: on mount, window focus, reconnect
        // Failed queries retry 3 times with exponential backoff
        // Inactive queries are GC'd after 5 mins (gcTime = 5 * 60 * 1000)
        // Structural sharing is enabled for efficient data comparison
      },
    },
  });
}

let browserQueryClient: QueryClient | undefined = undefined;

export function getQueryClient() {
  if (isServer) {
    // Server: create new client every time (so one user's data doesn't leak to another)
    return makeQueryClient();
  } else {
    // Client: reuse the same client (to keep cache + avoid bugs if React suspends)
    if (!browserQueryClient) browserQueryClient = makeQueryClient();
    return browserQueryClient;
  }
}
