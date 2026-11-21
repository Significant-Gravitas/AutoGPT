import { useGetV1GetSharedExecution } from "@/app/api/__generated__/endpoints/default/default";
import { okData } from "@/app/api/helpers";
import { isDeadLinkError, retryUnlessDeadLink } from "@/lib/share/dead-link";

export function useSharedExecutionPage(token: string) {
  const query = useGetV1GetSharedExecution(token, {
    query: { select: okData, retry: retryUnlessDeadLink },
  });

  return {
    executionData: query.data,
    // isPending, since a query paused offline is pending but not loading.
    isPending: query.isPending,
    isError: query.isError,
    // A 200 that carries no run is as dead as a 404.
    isDeadLink:
      isDeadLinkError(query.error) || (query.isSuccess && !query.data),
  };
}
