"use client";

import { ArrowSquareOutIcon, CreditCardIcon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "framer-motion";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

import { EASE_OUT } from "../../../helpers";
import { usePaymentMethodCard } from "./usePaymentMethodCard";

interface Props {
  index?: number;
}

export function PaymentMethodCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const { canManage, isOpening, onManage } = usePaymentMethodCard();

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
          Payment method
        </Text>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-4 rounded-[18px] border border-zinc-200 bg-white p-5 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[10px] bg-zinc-100 text-zinc-700">
            <CreditCardIcon size={20} />
          </div>
          <div className="flex min-w-0 flex-col">
            <Text variant="body-medium" as="span" className="text-textBlack">
              Manage payment method
            </Text>
            <Text variant="small" as="span" className="text-zinc-500">
              Open the Stripe portal to update your card or download invoices.
            </Text>
          </div>
        </div>

        <Button
          variant="secondary"
          size="small"
          onClick={onManage}
          disabled={!canManage}
          loading={isOpening}
        >
          Open portal
          <ArrowSquareOutIcon size={14} />
        </Button>
      </div>
    </motion.section>
  );
}
