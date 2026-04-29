"use client";

import { DownloadSimpleIcon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "framer-motion";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { EASE_OUT } from "../../../helpers";
import { useInvoicesCard } from "./useInvoicesCard";

interface Props {
  index?: number;
}

const STATUS_VARIANT: Record<string, "success" | "info" | "error"> = {
  paid: "success",
  open: "info",
  draft: "info",
  uncollectible: "error",
  void: "error",
};

export function InvoicesCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const { invoices, isLoading, isError, refetch } = useInvoicesCard();

  if (isLoading) {
    return <Skeleton className="h-[200px] rounded-[18px]" />;
  }

  if (isError) {
    return (
      <ErrorCard
        context="invoices"
        hint="We couldn't load your invoices."
        onRetry={() => void refetch()}
      />
    );
  }

  return (
    <motion.section
      initial={reduceMotion ? false : { opacity: 0, y: 12 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : { duration: 0.32, ease: EASE_OUT, delay: 0.04 + index * 0.05 }
      }
      className="flex w-full flex-col gap-2"
    >
      <div className="px-4">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Invoices
        </Text>
      </div>

      <div className="overflow-hidden rounded-[18px] border border-zinc-200 bg-white shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        {invoices.length === 0 ? (
          <div className="px-4 py-6">
            <Text variant="small" as="span" className="text-zinc-500">
              No invoices yet.
            </Text>
          </div>
        ) : (
          <table className="w-full border-collapse text-left">
            <thead className="bg-zinc-50">
              <tr className="border-b border-zinc-200">
                <Th>Invoice</Th>
                <Th>Date</Th>
                <Th>Description</Th>
                <Th align="right">Amount</Th>
                <Th>Status</Th>
                <th scope="col" className="w-12 px-4 py-3">
                  <span className="sr-only">Download</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {invoices.map((invoice, rowIndex) => (
                <tr
                  key={invoice.id}
                  className={
                    rowIndex !== invoices.length - 1
                      ? "border-b border-zinc-100 transition-colors hover:bg-zinc-50/60"
                      : "transition-colors hover:bg-zinc-50/60"
                  }
                >
                  <td className="px-4 py-4">
                    <Text
                      variant="body-medium"
                      as="span"
                      className="tabular-nums text-textBlack"
                    >
                      {invoice.number}
                    </Text>
                  </td>
                  <td className="px-4 py-4">
                    <Text variant="body" as="span" className="text-zinc-600">
                      {invoice.date}
                    </Text>
                  </td>
                  <td className="px-4 py-4">
                    <Text variant="body" as="span" className="text-zinc-600">
                      {invoice.description}
                    </Text>
                  </td>
                  <td className="px-4 py-4 text-right">
                    <Text
                      variant="body-medium"
                      as="span"
                      className="tabular-nums text-textBlack"
                    >
                      {invoice.amount}
                    </Text>
                  </td>
                  <td className="px-4 py-4">
                    <Badge
                      variant={STATUS_VARIANT[invoice.status] ?? "info"}
                      size="small"
                    >
                      {invoice.status}
                    </Badge>
                  </td>
                  <td className="px-4 py-4 text-right">
                    <Button
                      variant="secondary"
                      size="small"
                      aria-label={`Download invoice ${invoice.number}`}
                      className="h-7 min-w-0 px-1.5 py-0.5"
                      disabled={!invoice.pdfUrl}
                      onClick={() => {
                        if (invoice.pdfUrl) {
                          window.open(
                            invoice.pdfUrl,
                            "_blank",
                            "noopener,noreferrer",
                          );
                        }
                      }}
                    >
                      <DownloadSimpleIcon size={14} />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </motion.section>
  );
}

interface ThProps {
  children: React.ReactNode;
  align?: "left" | "right";
}

function Th({ children, align = "left" }: ThProps) {
  return (
    <th
      scope="col"
      className={`px-4 py-3 ${align === "right" ? "text-right" : "text-left"}`}
    >
      <Text
        variant="small-medium"
        as="span"
        className="uppercase tracking-[0.04em] text-zinc-500"
      >
        {children}
      </Text>
    </th>
  );
}
