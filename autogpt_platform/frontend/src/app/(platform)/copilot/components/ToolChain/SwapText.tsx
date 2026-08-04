"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ShimmerText } from "./ShimmerText";

const EASE_OUT_CUBIC = [0.33, 1, 0.68, 1] as const;

// Old label slides up + fades out, new one slides in from below.
export function SwapText({
  text,
  className,
  shimmer = false,
}: {
  text: string;
  className?: string;
  shimmer?: boolean;
}) {
  const reducedMotion = useReducedMotion();
  return (
    <span className={"inline-grid overflow-hidden " + (className ?? "")}>
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.span
          key={text}
          initial={reducedMotion ? false : { opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={reducedMotion ? undefined : { opacity: 0, y: -8 }}
          transition={{ duration: 0.18, ease: EASE_OUT_CUBIC }}
          className="block truncate"
        >
          {shimmer ? <ShimmerText text={text} /> : text}
        </motion.span>
      </AnimatePresence>
    </span>
  );
}

export function SwapIcon({
  swapKey,
  children,
}: {
  swapKey: string;
  children: React.ReactNode;
}) {
  const reducedMotion = useReducedMotion();
  return (
    <AnimatePresence mode="popLayout" initial={false}>
      <motion.span
        key={swapKey}
        initial={reducedMotion ? false : { opacity: 0, scale: 0.5 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={reducedMotion ? undefined : { opacity: 0, scale: 0.5 }}
        transition={{ duration: 0.15, ease: EASE_OUT_CUBIC }}
        className="flex"
      >
        {children}
      </motion.span>
    </AnimatePresence>
  );
}
