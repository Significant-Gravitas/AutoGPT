"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "framer-motion";

interface Props {
  position: string;
  label: string;
  text: string;
  isQuote: boolean;
  isHidden: boolean;
  // Null where the card should hold still: reduced motion, or a screen too
  // narrow to show the layer at all.
  drift: { durationSecs: number; delaySecs: number } | null;
  reduceMotion: boolean;
}

export function AmbientCard({
  position,
  label,
  text,
  isQuote,
  isHidden,
  drift,
  reduceMotion,
}: Props) {
  return (
    <motion.div
      className={cn(
        "absolute w-[196px] rounded-xl bg-surface-hover px-3 py-2.5 transition-opacity duration-500",
        position,
        isHidden && "opacity-0",
      )}
      animate={drift ? { y: [0, -6, 0] } : undefined}
      transition={
        drift
          ? {
              duration: drift.durationSecs,
              delay: drift.delaySecs,
              repeat: Infinity,
              ease: "easeInOut",
            }
          : undefined
      }
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={text}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: reduceMotion ? 0 : 0.4 }}
          className={cn(isQuote && "italic")}
        >
          <Text
            variant="small-semibold"
            as="span"
            className="block text-caption-sm uppercase tracking-eyebrow text-accent-strong"
          >
            {label}
          </Text>
          <Text
            variant="body"
            as="span"
            tone="secondary"
            className="block text-body-sm"
          >
            {text}
          </Text>
        </motion.div>
      </AnimatePresence>
    </motion.div>
  );
}
