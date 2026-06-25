"use client";

import { DownloadSimpleIcon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "framer-motion";

import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import {
  getSectionMotionProps,
  rowVariants,
  rowsContainerVariants,
} from "../../../helpers";
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
  const sectionMotion = getSectionMotionProps(index, Boolean(reduceMotion));

  if (isLoading) {
    return (
      <motion.div {...sectionMotion}>
        <Skeleton className="h-[200px] rounded-[18px]" />
      </motion.div>
    );
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
    <motion.section {...sectionMotion} className="flex w-full flex-col gap-2">
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
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead className="bg-zinc-50">
                <tr className="border-b border-zinc-200">
                  <Th>Invoice</Th>
                  <Th className="hidden sm:table-cell">Date</Th>
                  <Th className="hidden md:table-cell">Description</Th>
                  <Th align="right">Amount</Th>
                  <Th>Status</Th>
                  <th scope="col" className="w-12 px-2 py-3 sm:px-4">
                    <span className="sr-only">Download</span>
                  </th>
                </tr>
              </thead>
              <motion.tbody
                initial={reduceMotion ? false : "hidden"}
                animate="show"
                variants={reduceMotion ? undefined : rowsContainerVariants}
              >
                {invoices.map((invoice, rowIndex) => (
                  <motion.tr
                    key={invoice.id}
                    variants={reduceMotion ? undefined : rowVariants}
                    className={
                      rowIndex !== invoices.length - 1
                        ? "border-b border-zinc-100 transition-colors hover:bg-zinc-50/60"
                        : "transition-colors hover:bg-zinc-50/60"
                    }
                  >
                    <td className="px-2 py-4 sm:px-4">
                      <Text
                        variant="body-medium"
                        as="span"
                        className="tabular-nums text-textBlack"
                      >
                        {invoice.number}
                      </Text>
                    </td>
                    <td className="hidden px-2 py-4 sm:table-cell sm:px-4">
                      <Text variant="body" as="span" className="text-zinc-600">
                        {invoice.date}
                      </Text>
                    </td>
                    <td className="hidden px-2 py-4 md:table-cell md:px-4">
                      <Text variant="body" as="span" className="text-zinc-600">
                        {invoice.description}
                      </Text>
                    </td>
                    <td className="px-2 py-4 text-right sm:px-4">
                      <Text
                        variant="body-medium"
                        as="span"
                        className="tabular-nums text-textBlack"
                      >
                        {invoice.amount}
                      </Text>
                    </td>
                    <td className="px-2 py-4 sm:px-4">
                      <Badge
                        variant={STATUS_VARIANT[invoice.status] ?? "info"}
                        size="small"
                      >
                        {invoice.status}
                      </Badge>
                    </td>
                    <td className="px-2 py-4 text-right sm:px-4">
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
                  </motion.tr>
                ))}
              </motion.tbody>
            </table>
          </div>
        )}
      </div>
    </motion.section>
  );
}

interface ThProps {
  children: React.ReactNode;
  align?: "left" | "right";
  className?: string;
}

function Th({ children, align = "left", className = "" }: ThProps) {
  return (
    <th
      scope="col"
      className={`px-2 py-3 sm:px-4 ${align === "right" ? "text-right" : "text-left"} ${className}`}
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
