"use client";

import { useGetV1ListStripeInvoices } from "@/app/api/__generated__/endpoints/credits/credits";
import type { InvoiceListItem } from "@/app/api/__generated__/models/invoiceListItem";

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
  const { data, isLoading, isError, refetch } = useGetV1ListStripeInvoices(
    { limit: 24 },
    {
      query: {
        select: (res) => res.data as InvoiceListItem[] | undefined,
      },
    },
  );

  const invoices: InvoiceRow[] = (data ?? []).map((invoice, idx) => ({
    id: invoice.id,
    number: invoice.number ?? `#${idx + 1}`,
    date: formatShortDate(invoice.created_at),
    description: invoice.description ?? "Invoice",
    amount: formatCents(invoice.total_cents),
    status: invoice.status,
    hostedUrl: invoice.hosted_invoice_url ?? null,
    pdfUrl: invoice.invoice_pdf_url ?? null,
  }));

  return { invoices, isLoading, isError, refetch };
}
