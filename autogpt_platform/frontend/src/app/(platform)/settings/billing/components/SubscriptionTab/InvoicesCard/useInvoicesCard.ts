"use client";

// Until `pnpm generate:api` is re-run against the backend that ships the
// new `GET /credits/invoices` endpoint, render the Invoices list from the
// existing credit history (TOP_UP transactions). After regeneration, swap
// the implementation below for `useGetV1ListInvoices` — see
// `useInvoicesCard.next.ts` for the ready-to-paste version.
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
  const { data, isLoading } = useGetV1GetCreditHistory(
    { transaction_count_limit: 50 },
    {
      query: {
        select: (res) => res.data as TransactionHistory | undefined,
      },
    },
  );

  const invoices: InvoiceRow[] = (data?.transactions ?? [])
    .filter((tx) => tx.transaction_type === "TOP_UP")
    .map((tx) => ({
      id: tx.transaction_key ?? `${tx.transaction_time?.toString() ?? ""}`,
      number: tx.transaction_key ?? "—",
      date: formatShortDate(tx.transaction_time),
      description: tx.description ?? "Top up",
      amount: formatCents(Math.abs(tx.amount ?? 0)),
      status: "paid",
      hostedUrl: null,
      pdfUrl: null,
    }));

  return { invoices, isLoading };
}
