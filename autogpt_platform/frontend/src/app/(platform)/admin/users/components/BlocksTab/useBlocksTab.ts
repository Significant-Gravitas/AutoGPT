"use client";

import {
  getAdminListAccessBlocks,
  getGetAdminListAccessBlocksQueryKey,
  useGetAdminListAccessBlocks,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { useInfiniteQuery } from "@tanstack/react-query";
import { useState } from "react";
import { formatBlockedAddress } from "../../helpers";

export const BLOCKS_PAGE_SIZE = 50;

export type KindFilter = "all" | "IP" | "EMAIL_DOMAIN";

export function useBlocksTab() {
  const [search, setSearch] = useState("");
  const [kind, setKind] = useState<KindFilter>("all");
  // Fixed-size pages by offset, so any number of blocks stays reachable.
  const query = useInfiniteQuery({
    queryKey: [...getGetAdminListAccessBlocksQueryKey(), "pages"],
    queryFn: async ({ pageParam }) => {
      const page = okData(
        await getAdminListAccessBlocks({
          limit: BLOCKS_PAGE_SIZE,
          offset: pageParam,
        }),
      );
      if (!page) throw new Error("Failed to load the blocks");
      return page;
    },
    initialPageParam: 0,
    getNextPageParam: (last, pages) => {
      const loaded = pages.reduce((sum, page) => sum + page.blocks.length, 0);
      return loaded < last.total && last.blocks.length > 0 ? loaded : undefined;
    },
  });
  const blocks = query.data?.pages.flatMap((page) => page.blocks) ?? [];
  const needle = search.trim().toLowerCase();
  const visible = blocks.filter(
    (block) =>
      (kind === "all" || block.kind === kind) &&
      (!needle ||
        block.value.toLowerCase().includes(needle) ||
        formatBlockedAddress(block.value).includes(needle) ||
        block.reason.toLowerCase().includes(needle)),
  );

  return {
    blocks,
    visible,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
    hasMore: query.hasNextPage,
    showMore: () => void query.fetchNextPage(),
    search,
    setSearch,
    kind,
    setKind,
  };
}

// The count beside the tab.
export function useActiveBlockCount() {
  const query = useGetAdminListAccessBlocks(
    { limit: 1 },
    { query: { select: okData } },
  );
  return query.data?.total;
}
