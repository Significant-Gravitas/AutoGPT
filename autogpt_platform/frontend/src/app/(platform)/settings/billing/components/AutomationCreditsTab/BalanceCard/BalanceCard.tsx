"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";

import { formatCents, getSectionMotionProps } from "../../../helpers";
import { useBalanceCard } from "./useBalanceCard";

interface Props {
  index?: number;
}

const BALANCE_EXPLAINER =
  "Automation Credits are platform credits used for automation executions. They are not cash, cannot be withdrawn, and cannot be exchanged for subscription fees.";

export function BalanceCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const {
    balanceCents,
    isLoading,
    isError,
    refetch,
    open,
    setOpen,
    amount,
    setAmount,
    isValid,
    isAdding,
    handleSubmit,
  } = useBalanceCard();

  const sectionMotion = getSectionMotionProps(index, Boolean(reduceMotion));

  if (isLoading) {
    return (
      <motion.div {...sectionMotion}>
        <Skeleton className="h-[120px] rounded-[18px]" />
      </motion.div>
    );
  }

  if (isError || balanceCents === null) {
    return (
      <ErrorCard
        context="balance"
        hint="We couldn't load your credit balance."
        onRetry={() => void refetch()}
      />
    );
  }

  return (
    <motion.section {...sectionMotion} className="flex w-full flex-col gap-2">
      <div className="flex items-center gap-1 px-4">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Automation credits
        </Text>
        <InformationTooltip description={BALANCE_EXPLAINER} iconSize={22} />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-4 rounded-[18px] border border-zinc-200 bg-white p-5 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <Text variant="h2" as="span" className="text-textBlack">
          {formatCents(balanceCents)}
        </Text>
        <Dialog
          title="Add credits"
          styling={{ maxWidth: "420px" }}
          controlled={{ isOpen: open, set: setOpen }}
        >
          <Dialog.Trigger>
            <Button variant="primary" size="small">
              Add credits
            </Button>
          </Dialog.Trigger>
          <Dialog.Content>
            <div className="flex flex-col gap-3">
              <Text variant="small" as="span" className="text-zinc-500">
                We&apos;ll redirect you to Stripe to complete your purchase.
              </Text>
              <Input
                id="addcredits-amount"
                label="Amount (USD, whole dollars only, minimum $5)"
                type="amount"
                amountPrefix="$"
                decimalCount={0}
                placeholder="Amount"
                value={amount}
                onChange={(event) => setAmount(event.target.value)}
                wrapperClassName="!mb-0"
              />
            </div>
            <Dialog.Footer>
              <Button
                type="button"
                variant="ghost"
                size="small"
                onClick={() => setOpen(false)}
              >
                Cancel
              </Button>
              <Button
                type="button"
                variant="primary"
                size="small"
                disabled={!isValid || isAdding}
                loading={isAdding}
                onClick={handleSubmit}
              >
                Continue to checkout
              </Button>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog>
      </div>
    </motion.section>
  );
}
