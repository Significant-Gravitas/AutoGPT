"use client";

import { ArrowsClockwiseIcon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "framer-motion";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

import { EASE_OUT } from "../../../helpers";
import { AutoRefillDialog } from "./AutoRefillDialog";
import { useAutoRefillCard } from "./useAutoRefillCard";

interface Props {
  index?: number;
}

export function AutoRefillCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const {
    config,
    isEnabled,
    isLoading,
    isSaving,
    open,
    setOpen,
    threshold,
    setThreshold,
    refillAmount,
    setRefillAmount,
    isValid,
    save,
    disable,
  } = useAutoRefillCard();

  return (
    <motion.section
      initial={reduceMotion ? false : { opacity: 0, y: 12 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : { duration: 0.32, ease: EASE_OUT, delay: 0.04 + index * 0.05 }
      }
      className="flex w-full flex-wrap items-center justify-between gap-4 rounded-[18px] border border-zinc-200 bg-white p-5 shadow-[0_1px_2px_rgba(15,15,20,0.04)]"
    >
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[10px] bg-violet-100 text-violet-700">
          <ArrowsClockwiseIcon size={20} />
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <Text variant="body-medium" as="span" className="text-textBlack">
            Auto-refill
          </Text>
          <Text variant="body" as="span" className="text-zinc-500">
            {isEnabled && config
              ? `Refills $${(config.amount / 100).toFixed(0)} when balance drops below $${(config.threshold / 100).toFixed(0)}.`
              : "Top up automatically when your balance gets low."}
          </Text>
        </div>
      </div>

      <Button
        variant="secondary"
        size="small"
        onClick={() => setOpen(true)}
        disabled={isLoading}
      >
        {isEnabled ? "Edit" : "Set up auto-refill"}
      </Button>

      <AutoRefillDialog
        isOpen={open}
        onOpenChange={setOpen}
        threshold={threshold}
        setThreshold={setThreshold}
        refillAmount={refillAmount}
        setRefillAmount={setRefillAmount}
        isValid={isValid}
        isEnabled={isEnabled}
        isSaving={isSaving}
        onSave={save}
        onDisable={disable}
      />
    </motion.section>
  );
}
