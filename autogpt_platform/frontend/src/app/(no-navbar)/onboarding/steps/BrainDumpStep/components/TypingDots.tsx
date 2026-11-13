"use client";

import { motion } from "framer-motion";

const DOTS = [0, 1, 2];
const BLINK = { duration: 1, repeat: Infinity, ease: "easeInOut" } as const;

// Otto's "typing" before his reply shows. Decorative: the reply itself is
// already in the DOM for screen readers.
export function TypingDots() {
  return (
    <span
      aria-hidden="true"
      data-testid="otto-typing-dots"
      className="absolute top-3 left-0 flex gap-1.5"
    >
      {DOTS.map((index) => (
        <motion.span
          key={index}
          className="size-1.5 rounded-full bg-brand/50"
          initial={{ opacity: 0.3 }}
          animate={{ opacity: [0.3, 1, 0.3] }}
          transition={{ ...BLINK, delay: index * 0.15 }}
        />
      ))}
    </span>
  );
}
