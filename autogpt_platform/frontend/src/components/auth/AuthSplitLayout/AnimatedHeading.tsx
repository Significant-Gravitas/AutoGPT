"use client";

import { motion, useReducedMotion, type Variants } from "framer-motion";
import { ReactNode } from "react";

interface Props {
  lines: ReactNode[];
}

const lineVariants: Variants = {
  hidden: { opacity: 0, y: 24, filter: "blur(10px)" },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    filter: "blur(0px)",
    transition: {
      delay: 0.1 + i * 0.12,
      duration: 0.65,
      ease: [0.22, 1, 0.36, 1],
    },
  }),
};

const reducedVariants: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.2 } },
};

export function AnimatedHeading({ lines }: Props) {
  const prefersReducedMotion = useReducedMotion();
  const variants = prefersReducedMotion ? reducedVariants : lineVariants;

  return (
    <>
      {lines.map((line, i) => (
        <motion.span
          key={i}
          custom={i}
          initial="hidden"
          animate="visible"
          variants={variants}
          className="block"
        >
          {line}
        </motion.span>
      ))}
    </>
  );
}
