"use client";

// TODO: switch to the generated `useGetV1ListInvoices` hook once
// `pnpm generate:api` is re-run on this branch (the openapi.json change
// in this PR adds the `/credits/invoices` operation, but the regenerated
// client lives outside the diff). Until then, derive the invoice list
// from credit history TOP_UP transactions so the UI is still populated.
import { useGetV1GetCreditHistory } from "@/app/api/__generated__/endpoints/credits/credits";
import type { TransactionHistory } from "@/app/api/__generated__/models/transactionHistory";

import { formatCents, formatShortDate } from "../../../helpers";

export interface InvoiceRow {
  id: string;
  number: string;
  date: string;
  description: string;
  amount: string;
  status: string;
  hostedUrl: string | null;
  pdfUrl: string | null;
}

export function useInvoicesCard() {
  const { data, isLoading, isError, refetch } = useGetV1GetCreditHistory(
    { transaction_count_limit: 50 },
    {
      query: {
        select: (res) => res.data as TransactionHistory | undefined,
      },
    },
  );

  const invoices: InvoiceRow[] = (data?.transactions ?? [])
    .filter((tx) => tx.transaction_type === "TOP_UP")
    .map((tx, idx) => ({
      id: tx.transaction_key ?? `top-up-${idx}`,
      number: tx.transaction_key ?? "—",
      date: formatShortDate(tx.transaction_time),
      description: tx.description ?? "Top up",
      amount: formatCents(Math.abs(tx.amount ?? 0)),
      status: "paid",
      hostedUrl: null,
      pdfUrl: null,
    }));

  return { invoices, isLoading, isError, refetch };
}
