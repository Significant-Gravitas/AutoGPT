"use client";

import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "framer-motion";
import type { AmbientContent } from "./useAmbientPrompts";

interface Props {
  position: string;
  content: AmbientContent;
  isHidden: boolean;
  // Null where the card should hold still: paused, reduced motion, or a
  // screen too narrow to show the layer at all.
  drift: { durationSecs: number; delaySecs: number } | null;
  reduceMotion: boolean;
  onClick?: () => void;
}

const EASE = [0.16, 1, 0.3, 1] as const;

export function AmbientCard({
  position,
  content,
  isHidden,
  drift,
  reduceMotion,
  onClick,
}: Props) {
  return (
    <motion.div
      onClick={onClick}
      className={cn(
        "absolute w-[196px] rounded-xl bg-surface-hover px-3 py-2.5 transition-opacity duration-500",
        position,
        isHidden ? "opacity-0" : "pointer-events-auto cursor-pointer",
      )}
      animate={drift ? { y: [0, -7, 0, 5, 0] } : { y: 0 }}
      transition={
        drift
          ? {
              duration: drift.durationSecs,
              delay: drift.delaySecs,
              repeat: Infinity,
              ease: "easeInOut",
            }
          : { duration: 0.6, ease: EASE }
      }
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={content.text}
          initial={
            reduceMotion
              ? { opacity: 0 }
              : { opacity: 0, scale: 0.96, filter: "blur(6px)" }
          }
          animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
          exit={
            reduceMotion
              ? { opacity: 0 }
              : { opacity: 0, scale: 0.96, filter: "blur(6px)" }
          }
          transition={{ duration: reduceMotion ? 0 : 0.45, ease: EASE }}
          className={cn(content.isQuote && "italic")}
        >
          <span className="block font-sans text-caption-sm font-semibold tracking-eyebrow text-accent-strong uppercase">
            {content.label}
          </span>
          <span className="block font-sans text-body-sm font-normal text-foreground-secondary">
            {content.text}
          </span>
        </motion.div>
      </AnimatePresence>
    </motion.div>
  );
}
