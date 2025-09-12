import type { GraphExecutionsPaginated } from "@/app/api/__generated__/models/graphExecutionsPaginated";
import type { InfiniteData } from "@tanstack/react-query";

const AGENT_RUNNING_POLL_INTERVAL = 1500;

function hasValidExecutionsData(
  page: unknown,
): page is { data: GraphExecutionsPaginated } {
  return (
    typeof page === "object" &&
    page !== null &&
    "data" in page &&
    typeof (page as { data: unknown }).data === "object" &&
    (page as { data: unknown }).data !== null &&
    "executions" in (page as { data: GraphExecutionsPaginated }).data
  );
}

export function getRunsPollingInterval(
  pages: Array<unknown> | undefined,
  isRunsTab: boolean,
): number | false {
  if (!isRunsTab || !pages?.length) return false;

  try {
    const executions = pages.flatMap((page) => {
      if (!hasValidExecutionsData(page)) return [];
      return page.data.executions || [];
    });
    const hasActive = executions.some(
      (e) => e.status === "RUNNING" || e.status === "QUEUED",
    );
    return hasActive ? AGENT_RUNNING_POLL_INTERVAL : false;
  } catch {
    return false;
  }
}

export function computeRunsCount(
  infiniteData: InfiniteData<unknown> | undefined,
  runsLength: number,
): number {
  const lastPage = infiniteData?.pages.at(-1);
  if (!hasValidExecutionsData(lastPage)) return runsLength;
  return lastPage.data.pagination?.total_items || runsLength;
}

export function getNextRunsPageParam(lastPage: unknown): number | undefined {
  if (!hasValidExecutionsData(lastPage)) return undefined;

  const { pagination } = lastPage.data;
  const hasMore =
    pagination.current_page * pagination.page_size < pagination.total_items;
  return hasMore ? pagination.current_page + 1 : undefined;
}

export function extractRunsFromPages(
  infiniteData: InfiniteData<unknown> | undefined,
) {
  return (
    infiniteData?.pages.flatMap((page) => {
      if (!hasValidExecutionsData(page)) return [];
      return page.data.executions || [];
    }) || []
  );
}
