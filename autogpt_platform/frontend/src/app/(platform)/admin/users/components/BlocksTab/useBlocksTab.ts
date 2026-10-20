"use client";

import { useGetAdminListAccessBlocks } from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { useState } from "react";
import { formatBlockedAddress } from "../../helpers";

export const BLOCKS_PAGE_SIZE = 50;

export type KindFilter = "all" | "IP" | "EMAIL_DOMAIN";

export function useBlocksTab() {
  const [pages, setPages] = useState(1);
  const [search, setSearch] = useState("");
  const [kind, setKind] = useState<KindFilter>("all");
  const limit = BLOCKS_PAGE_SIZE * pages;
  const query = useGetAdminListAccessBlocks(
    { limit },
    { query: { select: okData, placeholderData: (previous) => previous } },
  );
  const blocks = query.data?.blocks ?? [];
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
    total: query.data?.total ?? 0,
    isLoading: query.isLoading,
    isError: !query.isLoading && !query.data,
    error: query.error,
    refetch: query.refetch,
    hasMore: blocks.length === limit,
    showMore: () => setPages(pages + 1),
    search,
    setSearch,
    kind,
    setKind,
  };
}

// The count beside the tab: the same first page the tab loads.
export function useActiveBlockCount() {
  const query = useGetAdminListAccessBlocks(
    { limit: BLOCKS_PAGE_SIZE },
    { query: { select: okData } },
  );
  return query.data?.total;
}
