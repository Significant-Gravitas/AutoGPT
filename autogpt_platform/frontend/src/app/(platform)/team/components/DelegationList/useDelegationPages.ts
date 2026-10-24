import {
  getListDelegationsQueryKey,
  listDelegations,
} from "@/app/api/__generated__/endpoints/experts/experts";
import type { DelegationListResponse } from "@/app/api/__generated__/models/delegationListResponse";
import type { ListDelegationsParams } from "@/app/api/__generated__/models/listDelegationsParams";
import { keepPreviousData, useInfiniteQuery } from "@tanstack/react-query";
import { useState } from "react";
import type { DelegationFilter } from "./helpers";

const PAGE_SIZE = 50;

interface Scope {
  expertId?: string;
}

/** Hand-offs a page at a time, filtered by chip on the server: a match older
 *  than the first page is still one "Show more" away, and the summary counts
 *  every hand-off in scope whatever the filter. */
export function useDelegationPages({ expertId }: Scope = {}) {
  const [filter, setFilter] = useState<DelegationFilter>("all");
  const params: ListDelegationsParams = {
    ...(expertId ? { expert_id: expertId } : {}),
    ...(filter === "all" ? {} : { group: filter }),
  };
  const query = useInfiniteQuery({
    // Under the generated key, so invalidating the list refreshes the pages.
    queryKey: [...getListDelegationsQueryKey(params), "pages"],
    initialPageParam: 0,
    queryFn: ({ pageParam }) => fetchPage(params, pageParam),
    getNextPageParam: (last, pages) => {
      const loaded = pages.reduce((n, page) => n + page.delegations.length, 0);
      return loaded < last.total ? loaded : undefined;
    },
    placeholderData: keepPreviousData,
  });
  const pages = query.data?.pages ?? [];

  return {
    filter,
    setFilter,
    delegations: pages.flatMap((page) => page.delegations),
    summary: pages[0]?.summary ?? null,
    hasMore: query.hasNextPage,
    loadMore: () => void query.fetchNextPage(),
    isLoadingMore: query.isFetchingNextPage,
    isLoading: query.isLoading,
    isError: query.isError && !query.data,
    refetch: query.refetch,
  };
}

async function fetchPage(
  params: ListDelegationsParams,
  offset: number,
): Promise<DelegationListResponse> {
  const res = await listDelegations({ ...params, limit: PAGE_SIZE, offset });
  if (res.status !== 200) throw new Error("Could not load the hand-offs");
  return res.data;
}
