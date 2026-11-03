import { motion, useIsPresent, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";

interface Props {
  id: string;
  labelledBy: string;
  children: ReactNode;
}

const EASE = [0.22, 1, 0.36, 1] as const;

export function FoldPanel({ id, labelledBy, children }: Props) {
  const present = useIsPresent();
  const reduceMotion = useReducedMotion();
  return (
    <motion.div
      id={id}
      role="region"
      aria-labelledby={labelledBy}
      inert={!present}
      initial={reduceMotion ? false : { height: 0, opacity: 0 }}
      animate={{ height: "auto", opacity: 1 }}
      exit={reduceMotion ? undefined : { height: 0, opacity: 0 }}
      transition={{
        height: { duration: reduceMotion ? 0 : 0.44, ease: EASE },
        opacity: { duration: reduceMotion ? 0 : 0.3, ease: EASE },
      }}
      className="overflow-hidden"
    >
      {children}
    </motion.div>
  );
}
