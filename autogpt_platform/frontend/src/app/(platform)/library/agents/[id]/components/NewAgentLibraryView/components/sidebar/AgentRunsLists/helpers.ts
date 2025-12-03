import type { GraphExecutionsPaginated } from "@/app/api/__generated__/models/graphExecutionsPaginated";
import { Pagination } from "@/app/api/__generated__/models/pagination";
import type { InfiniteData } from "@tanstack/react-query";

const AGENT_RUNNING_POLL_INTERVAL = 1500;

function hasValidExecutionsData(
  page: unknown,
): page is { data: GraphExecutionsPaginated } {
  return (
    typeof page === "object" &&
    page !== null &&
    "data" in page &&
    typeof page.data === "object" &&
    page.data !== null &&
    "executions" in page.data
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

export function getPaginatedTotalCount(
  infiniteData: InfiniteData<unknown> | undefined,
  runsLength: number,
): number {
  const lastPage = infiniteData?.pages.at(-1);
  if (!hasValidPaginationInfo(lastPage)) return runsLength;
  return lastPage.data.pagination?.total_items || runsLength;
}

export function getPaginationNextPageNumber(
  lastPage:
    | { data: { pagination?: Pagination; [key: string]: any } }
    | undefined,
): number | undefined {
  if (!hasValidPaginationInfo(lastPage)) return undefined;

  const { pagination } = lastPage.data;
  const hasMore =
    pagination.current_page * pagination.page_size < pagination.total_items;
  return hasMore ? pagination.current_page + 1 : undefined;
}

export function unpaginate<
  TItemData extends object,
  TPageDataKey extends string,
>(
  infiniteData: InfiniteData<{
    status: number;
    data: { [key in TPageDataKey]: TItemData[] } | Record<string, any>;
  }>,
  pageListKey: TPageDataKey &
    keyof (typeof infiniteData)["pages"][number]["data"],
): TItemData[] {
  return (
    infiniteData?.pages.flatMap((page) => {
      if (!hasValidListPage<TItemData, TPageDataKey>(page, pageListKey))
        return [];
      return page.data[pageListKey] || [];
    }) || []
  );
}

function hasValidListPage<TItemData extends object, TKey extends string>(
  page: unknown,
  pageListKey: TKey,
): page is { data: { [key in TKey]: TItemData[] } } {
  return (
    typeof page === "object" &&
    page !== null &&
    "data" in page &&
    typeof page.data === "object" &&
    page.data !== null &&
    pageListKey in page.data &&
    Array.isArray((page.data as Record<string, unknown>)[pageListKey])
  );
}

function hasValidPaginationInfo(
  page: unknown,
): page is { data: { pagination: Pagination; [key: string]: any } } {
  return (
    typeof page === "object" &&
    page !== null &&
    "data" in page &&
    typeof page.data === "object" &&
    page.data !== null &&
    "pagination" in page.data &&
    typeof page.data.pagination === "object" &&
    page.data.pagination !== null &&
    "total_items" in page.data.pagination &&
    "total_pages" in page.data.pagination &&
    "current_page" in page.data.pagination &&
    "page_size" in page.data.pagination
  );
}
