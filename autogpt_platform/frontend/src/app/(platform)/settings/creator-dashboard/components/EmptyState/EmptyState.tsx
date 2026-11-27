"use client";
import { motion, useReducedMotion } from "framer-motion";

import { Text } from "@/components/atoms/Text/Text";

import { Rocket01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { duration, ease } from "@/components/styles/motion";

export function EmptyState() {
  const reduceMotion = useReducedMotion();

  return (
    <motion.div
      initial={reduceMotion ? false : { opacity: 0, y: 8 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : { duration: duration.slow, ease: ease.emphasized }
      }
      className="flex flex-col items-center justify-center gap-3 px-6 py-12 text-center text-foreground-subtle"
    >
      <Icon icon={Rocket01Icon} size={28} />
      <Text variant="body" weight="medium">
        No submissions yet
      </Text>
      <Text variant="small" tone="muted" className="max-w-[460px]">
        Once you submit a workflow to the store, it&apos;ll show up here with
        its status, runs, and reviews.
      </Text>
    </motion.div>
  );
}
