import { useGetV1GetSharedExecution } from "@/app/api/__generated__/endpoints/default/default";
import { okData } from "@/app/api/helpers";

// The backend answers 404 for a link that was never shared, has been unshared
// or whose run was deleted, and 422 for a token it can't read at all. Either
// way the link is dead, and asking again won't change the answer.
const DEAD_LINK_STATUSES = [404, 422];

function isDeadLinkError(error: unknown) {
  const status = (error as { status?: unknown } | null)?.status;
  return typeof status === "number" && DEAD_LINK_STATUSES.includes(status);
}

function retryUnlessDeadLink(failureCount: number, error: unknown) {
  return !isDeadLinkError(error) && failureCount < 3;
}

export function useSharedExecutionPage(token: string) {
  const query = useGetV1GetSharedExecution(token, {
    query: { select: okData, retry: retryUnlessDeadLink },
  });

  return {
    executionData: query.data,
    isLoading: query.isPending,
    isError: query.isError,
    isDeadLink: isDeadLinkError(query.error),
  };
}
