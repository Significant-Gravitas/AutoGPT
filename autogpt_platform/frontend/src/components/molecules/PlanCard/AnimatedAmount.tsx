"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { cn } from "@/lib/utils";

interface Props {
  value: string;
  className?: string;
}

const EASE_OUT = [0.22, 1, 0.36, 1] as const;

export function AnimatedAmount({ value, className }: Props) {
  const reduceMotion = useReducedMotion();

  if (reduceMotion) {
    return <span className={cn("inline-block", className)}>{value}</span>;
  }

  return (
    <AnimatePresence mode="popLayout" initial={false}>
      <motion.span
        key={value}
        className={cn("inline-block", className)}
        initial={{ y: 6, opacity: 0, filter: "blur(2px)" }}
        animate={{ y: 0, opacity: 1, filter: "blur(0px)" }}
        exit={{ y: -6, opacity: 0, filter: "blur(2px)" }}
        transition={{ duration: 0.22, ease: EASE_OUT }}
      >
        {value}
      </motion.span>
    </AnimatePresence>
  );
}
