"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ReactNode } from "react";

interface Props {
  swapKey: string;
  children: ReactNode;
  className?: string;
}

// Swaps one piece of content for another in place: the outgoing content
// lifts and blurs away before the incoming one settles, so a state change
// reads as a change rather than a re-render.
export function SwapFade({ swapKey, children, className }: Props) {
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
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        key={swapKey}
        initial={states.initial}
        animate={states.animate}
        exit={states.exit}
        transition={{ duration: 0.22, ease: [0.32, 0.72, 0, 1] }}
        className={className}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
