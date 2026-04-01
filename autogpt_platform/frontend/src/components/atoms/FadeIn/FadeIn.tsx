"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

interface Props {
  children: ReactNode;
  onComplete?: () => void;
}

export function FadeIn({ children, onComplete }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      onAnimationComplete={onComplete}
    >
      {children}
    </motion.div>
  );
}
