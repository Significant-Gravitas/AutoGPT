export type AsyncStatus = "loading" | "error" | "loaded";

export function queryToAsyncStatus<T>(
  query: {
    data: T | undefined;
    enabled: boolean;
    isError: boolean;
    isFetching: boolean;
  },
  options: { keepCachedData: boolean },
): AsyncStatus {
  if (!query.enabled) return "loaded";
  if (options.keepCachedData && query.data !== undefined) return "loaded";
  if (query.isFetching) return "loading";
  if (query.isError) return "error";
  return query.data !== undefined ? "loaded" : "loading";
}
