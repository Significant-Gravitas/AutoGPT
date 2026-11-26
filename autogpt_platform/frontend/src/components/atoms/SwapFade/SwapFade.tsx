"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ReactNode } from "react";
import { duration, ease } from "@/components/styles/motion";

interface Props {
  swapKey: string;
  children: ReactNode;
  className?: string;
  mode?: "sync" | "wait" | "popLayout";
}

// Swaps one piece of content for another in place: the outgoing content
// lifts and blurs away before the incoming one settles, so a state change
// reads as a change rather than a re-render.
export function SwapFade({
  swapKey,
  children,
  className,
  mode = "wait",
}: Props) {
  const prefersReducedMotion = useReducedMotion();

  const states = prefersReducedMotion
    ? {
        initial: { opacity: 0 },
        animate: { opacity: 1 },
        exit: { opacity: 0 },
      }
    : {
        initial: { opacity: 0, y: 8, filter: "blur(4px)" },
        animate: { opacity: 1, y: 0, filter: "blur(0px)" },
        exit: { opacity: 0, y: -8, filter: "blur(4px)" },
      };

  return (
    <AnimatePresence mode={mode} initial={false}>
      <motion.div
        key={swapKey}
        data-slot="swap-fade"
        initial={states.initial}
        animate={states.animate}
        exit={states.exit}
        transition={{ duration: duration.base, ease: ease.drawer }}
        className={className}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
