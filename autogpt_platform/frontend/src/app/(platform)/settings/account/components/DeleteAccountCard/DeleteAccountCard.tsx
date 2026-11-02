"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { DeleteAccountFlow } from "@/components/organisms/DeleteAccountFlow/DeleteAccountFlow";
import { useDeleteAccountFlowOpener } from "@/components/organisms/DeleteAccountFlow/useDeleteAccountFlowOpener";

import { EASE_OUT } from "../../helpers";

interface Props {
  index?: number;
}

export function DeleteAccountCard({ index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const { isFlowOpen, openFlow, handleFlowOpenChange } =
    useDeleteAccountFlowOpener("settings");

  return (
    <motion.section
      initial={reduceMotion ? false : { opacity: 0, y: 12 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : {
              duration: 0.32,
              ease: EASE_OUT,
              delay: 0.04 + index * 0.05,
            }
      }
      className="flex w-full flex-col"
    >
      <div className="flex flex-col items-start gap-3 rounded-large border border-border-base bg-surface px-4 py-4 shadow-message-table sm:flex-row sm:items-center sm:justify-between sm:gap-4">
        <div className="flex min-w-0 flex-col gap-0.5">
          <Text variant="body-medium" as="span">
            Delete account
          </Text>
          <Text variant="small" as="span" tone="muted">
            Close your account, disconnect your apps and cancel your plan.
            Everything in it is erased after 14 days.
          </Text>
        </div>

        <Button
          type="button"
          size="small"
          variant="secondary"
          onClick={openFlow}
          // oxlint-disable-next-line design-system/no-restyle -- Button has no danger-label variant; the red marks the destructive action, in Text's danger colour
          className="shrink-0 text-error-hover"
        >
          Delete account
        </Button>
      </div>

      <DeleteAccountFlow
        source="settings"
        open={isFlowOpen}
        onOpenChange={handleFlowOpenChange}
      />
    </motion.section>
  );
}
