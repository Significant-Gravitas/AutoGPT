import { isServer, QueryClient } from "@tanstack/react-query";

function makeQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // Increase stale time to 5 minutes for better caching
        staleTime: 5 * 60 * 1000, // 5 minutes

        // Keep data in cache for 30 minutes (was 5 minutes default)
        gcTime: 30 * 60 * 1000, // 30 minutes

        // Reduce refetch frequency
        refetchOnWindowFocus: false,
        refetchOnReconnect: "always",

        // Retry configuration
        retry: 2, // Reduce from 3 to 2 retries
        retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),

        // Highlighting some important defaults to avoid confusion
        // Queries are stale by default → triggers background refetch
        // Failed queries retry with exponential backoff
        // Structural sharing is enabled for efficient data comparison
        // For more info, visit https://tanstack.com/query/latest/docs/framework/react/guides/important-defaults
      },
      mutations: {
        // Mutation defaults
        retry: 1,
        retryDelay: 1000,
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
