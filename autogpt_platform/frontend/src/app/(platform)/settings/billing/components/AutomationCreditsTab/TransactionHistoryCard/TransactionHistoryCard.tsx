"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useId } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { getSectionMotionProps } from "../../../helpers";
import { TransactionHistoryTable } from "./components/TransactionHistoryTable";
import { useTransactionHistoryCard } from "./useTransactionHistoryCard";

type Props = { index?: number };

export function TransactionHistoryCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const history = useTransactionHistoryCard();
  const headingID = useId();

  async function retryHistory() {
    await history.refetch();
    document.getElementById(headingID)?.focus({ preventScroll: true });
  }
  return (
    <motion.section
      {...getSectionMotionProps(index, Boolean(reduceMotion))}
      className="flex w-full flex-col gap-3"
      aria-label="Transaction history"
    >
      <div className="flex flex-wrap items-end justify-between gap-2 px-4">
        <div>
          <Text
            variant="body-medium"
            as="h2"
            id={headingID}
            tabIndex={-1}
            className="focus:outline-none"
          >
            Transaction history
          </Text>
          <Text variant="small" className="mt-1 text-zinc-600">
            Changes to your automation credit balance.
          </Text>
        </div>
        <Text variant="small" className="text-zinc-600">
          USD · Your local time
        </Text>
      </div>
      {history.isLoading ? (
        <HistorySkeleton />
      ) : history.isError ? (
        <ErrorCard
          context="transaction history"
          hint="We couldn’t load your transactions. Please try again."
          onRetry={() => void retryHistory()}
        />
      ) : history.transactions.length === 0 ? (
        <EmptyHistory />
      ) : (
        <TransactionHistoryTable
          transactions={history.transactions}
          hasMore={history.hasMore}
          isLoadingMore={history.isLoadingMore}
          isLoadMoreError={history.isLoadMoreError}
          onLoadMore={() => history.loadMore()}
        />
      )}
      {history.isRefreshError && (
        <div
          className="flex flex-wrap items-center justify-between gap-2 px-4"
          role="alert"
        >
          <Text variant="small" className="text-zinc-600">
            History couldn’t be refreshed. Showing previously loaded activity.
          </Text>
          <Button
            variant="ghost"
            size="small"
            onClick={() => void retryHistory()}
          >
            Try again
          </Button>
        </div>
      )}
      <Text variant="small" className="px-4 text-zinc-600">
        Each agent run combines its recorded charges and adjustments. Only runs
        with credit activity appear here.
      </Text>
    </motion.section>
  );
}

function HistorySkeleton() {
  return (
    <div
      role="status"
      aria-label="Loading transaction history"
      className="rounded-large border border-zinc-200 bg-white p-5"
    >
      <div className="space-y-6">
        {Array.from({ length: 5 }, (_, index) => (
          <div key={index} className="flex items-center justify-between gap-6">
            <Skeleton className="h-5 w-40" />
            <Skeleton className="h-5 w-16" />
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyHistory() {
  return (
    <div className="rounded-large border border-zinc-200 bg-white px-5 py-10 text-center">
      <Text variant="body-medium">No transactions yet.</Text>
      <Text variant="small" className="mt-2 text-zinc-600">
        Credit purchases and paid activity will appear here.
      </Text>
    </div>
  );
}
