"use client";

import {
  getGetV1GetCreditHistoryQueryKey,
  getV1GetCreditHistory,
} from "@/app/api/__generated__/endpoints/credits/credits";
import type { TransactionHistory } from "@/app/api/__generated__/models/transactionHistory";
import { useOrgTeamStore } from "@/services/org-team/store";
import { useInfiniteQuery } from "@tanstack/react-query";
import type { Transaction } from "./helpers";

export function useTransactionHistoryCard() {
  const activeOrgID = useOrgTeamStore((state) => state.activeOrgID);
  const history = useInfiniteQuery({
    queryKey: [...getGetV1GetCreditHistoryQueryKey(), "receipts", activeOrgID],
    initialPageParam: undefined as string | undefined,
    queryFn: async ({ pageParam, signal }) => {
      const response = await getV1GetCreditHistory(
        { transaction_count_limit: 25, cursor: pageParam },
        { signal },
      );
      if (response.status !== 200)
        throw new Error("Transaction history could not be loaded");
      return response.data;
    },
    getNextPageParam: (page) => page.next_cursor || undefined,
    retry: false,
  });
  const transactions = mergeTransactions(history.data?.pages ?? []);
  return {
    transactions,
    isLoading: history.isLoading,
    isError: history.isError && !history.data,
    isLoadingMore: history.isFetchingNextPage,
    isLoadMoreError: history.isFetchNextPageError,
    isRefreshError: history.isRefetchError && !history.isFetchNextPageError,
    hasMore: !!history.hasNextPage,
    refetch: history.refetch,
    loadMore: history.fetchNextPage,
  };
}

function mergeTransactions(pages: TransactionHistory[]): Transaction[] {
  const transactions = new Map<string, Transaction>();
  for (const page of pages) {
    for (const item of page.transactions) {
      const id =
        item.id ||
        item.transaction_key ||
        `execution:${item.usage_execution_id}`;
      if (!transactions.has(id))
        transactions.set(id, {
          ...item,
          id,
          amount: item.amount ?? 0,
          transaction_type: item.transaction_type ?? "USAGE",
          receipt_as_of: page.snapshot_at,
        });
    }
  }
  return Array.from(transactions.values());
}
