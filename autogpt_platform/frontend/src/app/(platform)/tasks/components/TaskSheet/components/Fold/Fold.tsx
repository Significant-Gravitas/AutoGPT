"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";

interface Props {
  open: boolean;
  id?: string;
  children: ReactNode;
}

const EASE = [0.22, 1, 0.36, 1] as const;

/** Folds its content open and shut (height + fade); instant under reduced motion. */
export function Fold({ open, id, children }: Props) {
  const reduceMotion = useReducedMotion();

  return (
    <AnimatePresence initial={false}>
      {open ? (
        <motion.div
          id={id}
          key="fold"
          initial={reduceMotion ? false : { height: 0, opacity: 0.3 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={reduceMotion ? { opacity: 0 } : { height: 0, opacity: 0.3 }}
          transition={
            reduceMotion ? { duration: 0 } : { duration: 0.44, ease: EASE }
          }
          className="overflow-hidden"
        >
          {children}
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
