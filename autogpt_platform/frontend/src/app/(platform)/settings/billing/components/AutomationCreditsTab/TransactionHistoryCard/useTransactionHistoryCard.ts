"use client";

import { useGetV1GetCreditHistory } from "@/app/api/__generated__/endpoints/credits/credits";
import type { TransactionHistory } from "@/app/api/__generated__/models/transactionHistory";

import { formatCents } from "../../../helpers";

export interface TransactionRow {
  id: string;
  date: string;
  description: string;
  amount: string;
  balance: string;
  kind: "credit" | "debit";
}

export function useTransactionHistoryCard() {
  const { data, isLoading, isError, refetch } = useGetV1GetCreditHistory(
    { transaction_count_limit: 50 },
    {
      query: {
        select: (res) => res.data as TransactionHistory | undefined,
      },
    },
  );

  const transactions: TransactionRow[] = (data?.transactions ?? []).map(
    (tx, idx) => {
      const amountCents = tx.amount ?? 0;
      return {
        id:
          tx.transaction_key ??
          (tx.transaction_time
            ? `${tx.transaction_time.toString()}-${idx}`
            : `tx-${idx}`),
        date: tx.transaction_time
          ? new Date(tx.transaction_time).toLocaleDateString(undefined, {
              month: "short",
              day: "numeric",
              year: "numeric",
            })
          : "—",
        description: tx.description ?? tx.transaction_type ?? "Transaction",
        amount: `${amountCents > 0 ? "+" : ""}${formatCents(amountCents)}`,
        balance: formatCents(tx.running_balance ?? 0),
        kind: amountCents >= 0 ? "credit" : "debit",
      };
    },
  );

  return { transactions, isLoading, isError, refetch };
}
