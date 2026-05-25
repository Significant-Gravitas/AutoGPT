"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import {
  getSectionMotionProps,
  rowVariants,
  rowsContainerVariants,
} from "../../../helpers";
import { useTransactionHistoryCard } from "./useTransactionHistoryCard";

interface Props {
  index?: number;
}

export function TransactionHistoryCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const { transactions, isLoading, isError, refetch } =
    useTransactionHistoryCard();
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
        context="transaction history"
        hint="We couldn't load your recent transactions."
        onRetry={() => void refetch()}
      />
    );
  }

  return (
    <motion.section {...sectionMotion} className="flex w-full flex-col gap-2">
      <div className="px-4">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Transaction history
        </Text>
      </div>

      <div className="overflow-hidden rounded-[18px] border border-zinc-200 bg-white shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        {transactions.length === 0 ? (
          <div className="px-4 py-6">
            <Text variant="small" as="span" className="text-zinc-500">
              No transactions yet.
            </Text>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead className="bg-zinc-50">
                <tr className="border-b border-zinc-200">
                  <Th className="hidden sm:table-cell">Date</Th>
                  <Th>Description</Th>
                  <Th align="right">Amount</Th>
                  <Th align="right" className="hidden sm:table-cell">
                    Balance
                  </Th>
                </tr>
              </thead>
              <motion.tbody
                initial={reduceMotion ? false : "hidden"}
                animate="show"
                variants={reduceMotion ? undefined : rowsContainerVariants}
              >
                {transactions.map((transaction, rowIndex) => (
                  <motion.tr
                    key={transaction.id}
                    variants={reduceMotion ? undefined : rowVariants}
                    className={
                      rowIndex !== transactions.length - 1
                        ? "border-b border-zinc-100 transition-colors hover:bg-zinc-50/60"
                        : "transition-colors hover:bg-zinc-50/60"
                    }
                  >
                    <td className="hidden px-2 py-4 sm:table-cell sm:px-4">
                      <Text variant="body" as="span" className="text-zinc-600">
                        {transaction.date}
                      </Text>
                    </td>
                    <td className="px-2 py-4 sm:px-4">
                      <Text variant="body" as="span" className="text-textBlack">
                        {transaction.description}
                      </Text>
                      <Text
                        variant="small"
                        as="span"
                        className="block text-zinc-500 sm:hidden"
                      >
                        {transaction.date}
                      </Text>
                    </td>
                    <td className="px-2 py-4 text-right sm:px-4">
                      <Text
                        variant="body-medium"
                        as="span"
                        className={
                          transaction.kind === "credit"
                            ? "tabular-nums text-emerald-700"
                            : "tabular-nums text-red-600"
                        }
                      >
                        {transaction.amount}
                      </Text>
                    </td>
                    <td className="hidden px-2 py-4 text-right sm:table-cell sm:px-4">
                      <Text
                        variant="body-medium"
                        as="span"
                        className="tabular-nums text-textBlack"
                      >
                        {transaction.balance}
                      </Text>
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
