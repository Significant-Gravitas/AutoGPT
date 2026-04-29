"use client";

import { useGetV1GetCreditHistory } from "@/app/api/__generated__/endpoints/credits/credits";
import type { TransactionHistory } from "@/app/api/__generated__/models/transactionHistory";

export interface DailyUsageRow {
  day: string;
  date: string;
  amount: number;
  runs: number;
}

const DAY_MS = 24 * 60 * 60 * 1000;
const WINDOW_DAYS = 30;

export function useUsageCard() {
  const { data, isLoading } = useGetV1GetCreditHistory(
    { transaction_count_limit: 200 },
    {
      query: {
        select: (res) => res.data as TransactionHistory | undefined,
      },
    },
  );

  const transactions = data?.transactions ?? [];
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const buckets = new Map<
    string,
    { day: string; date: string; amountCents: number; runs: number }
  >();

  for (let i = WINDOW_DAYS - 1; i >= 0; i--) {
    const d = new Date(today.getTime() - i * DAY_MS);
    buckets.set(d.toISOString().slice(0, 10), {
      day: d.toLocaleDateString(undefined, { weekday: "short" }),
      date: d.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
      }),
      amountCents: 0,
      runs: 0,
    });
  }

  transactions.forEach((tx) => {
    if (tx.transaction_type !== "USAGE") return;
    if (!tx.transaction_time) return;
    const txDate = new Date(tx.transaction_time);
    const key = txDate.toISOString().slice(0, 10);
    const bucket = buckets.get(key);
    if (!bucket) return;
    bucket.amountCents += Math.abs(tx.amount ?? 0);
    bucket.runs += 1;
  });

  const usage: DailyUsageRow[] = Array.from(buckets.values()).map((b) => ({
    day: b.day,
    date: b.date,
    amount: b.amountCents / 100,
    runs: b.runs,
  }));

  const hasUsage = usage.some((u) => u.amount > 0 || u.runs > 0);

  return { usage, hasUsage, isLoading };
}
