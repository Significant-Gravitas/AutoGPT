"use client";

import { motion, useReducedMotion } from "framer-motion";
import { ReactNode } from "react";
import { duration, ease } from "@/components/styles/motion";

interface Props {
  children: ReactNode;
  onComplete?: () => void;
  className?: string;
}

export function FadeIn({ children, onComplete, className }: Props) {
  const prefersReducedMotion = useReducedMotion();

  const states = prefersReducedMotion
    ? { initial: { opacity: 0 }, animate: { opacity: 1 } }
    : { initial: { opacity: 0, y: 12 }, animate: { opacity: 1, y: 0 } };

  return (
    <motion.div
      data-slot="fade-in"
      initial={states.initial}
      animate={states.animate}
      transition={{ duration: duration.slower, ease: ease.out }}
      onAnimationComplete={onComplete}
      className={className}
    >
      {children}
    </motion.div>
  );
}
