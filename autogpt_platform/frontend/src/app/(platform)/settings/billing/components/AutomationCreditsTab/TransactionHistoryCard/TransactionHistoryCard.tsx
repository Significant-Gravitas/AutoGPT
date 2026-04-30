"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { EASE_OUT } from "../../../helpers";
import { useTransactionHistoryCard } from "./useTransactionHistoryCard";

interface Props {
  index?: number;
}

export function TransactionHistoryCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const { transactions, isLoading, isError, refetch } =
    useTransactionHistoryCard();

  if (isLoading) {
    return <Skeleton className="h-[200px] rounded-[18px]" />;
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
          <table className="w-full border-collapse text-left">
            <thead className="bg-zinc-50">
              <tr className="border-b border-zinc-200">
                <Th>Date</Th>
                <Th>Description</Th>
                <Th align="right">Amount</Th>
                <Th align="right">Balance</Th>
              </tr>
            </thead>
            <tbody>
              {transactions.map((transaction, rowIndex) => (
                <tr
                  key={transaction.id}
                  className={
                    rowIndex !== transactions.length - 1
                      ? "border-b border-zinc-100 transition-colors hover:bg-zinc-50/60"
                      : "transition-colors hover:bg-zinc-50/60"
                  }
                >
                  <td className="px-4 py-4">
                    <Text variant="body" as="span" className="text-zinc-600">
                      {transaction.date}
                    </Text>
                  </td>
                  <td className="px-4 py-4">
                    <Text variant="body" as="span" className="text-textBlack">
                      {transaction.description}
                    </Text>
                  </td>
                  <td className="px-4 py-4 text-right">
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
                  <td className="px-4 py-4 text-right">
                    <Text
                      variant="body-medium"
                      as="span"
                      className="tabular-nums text-textBlack"
                    >
                      {transaction.balance}
                    </Text>
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
