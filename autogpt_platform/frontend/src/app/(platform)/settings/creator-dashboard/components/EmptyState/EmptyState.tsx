"use client";
import { motion, useReducedMotion } from "framer-motion";

import { Text } from "@/components/atoms/Text/Text";

import { EASE_OUT } from "../../helpers";
import { Rocket01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function EmptyState() {
  const reduceMotion = useReducedMotion();

  return (
    <motion.div
      initial={reduceMotion ? false : { opacity: 0, y: 8 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={reduceMotion ? undefined : { duration: 0.28, ease: EASE_OUT }}
      className="flex flex-col items-center justify-center gap-3 px-6 py-12 text-center"
    >
      <Icon icon={Rocket01Icon} size={28} className="text-zinc-400" />
      <Text variant="body-medium" className="text-textBlack">
        No submissions yet
      </Text>
      <Text variant="small" className="max-w-[460px] text-zinc-500">
        Once you submit an agent to the store, it&apos;ll show up here with its
        status, runs, and reviews.
      </Text>
    </motion.div>
  );
}
