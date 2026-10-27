import { cn } from "@/lib/utils";
import { motion, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";

interface Props {
  labelledBy: string;
  twoColumns: boolean;
  children: ReactNode;
}

// The step's one card: the plan beside the trial on wide screens, the plan
// first and the trial under it on narrow ones.
export function PlanStepCard({ labelledBy, twoColumns, children }: Props) {
  const reduceMotion = useReducedMotion();
  return (
    <motion.div
      initial={reduceMotion ? false : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="w-full px-4 pt-10 lg:pt-0"
    >
      <section
        aria-labelledby={labelledBy}
        className={cn(
          // Clipped rather than hidden, so the narrow layout's sticky
          // checkout still sticks to the viewport.
          "mx-auto grid w-full overflow-clip rounded-[1.75rem] border border-border-base/80 bg-surface shadow-[0_1px_2px_rgba(24,24,27,0.04),0_32px_64px_-32px_rgba(24,24,27,0.18)]",
          twoColumns ? "max-w-[64rem] lg:grid-cols-2" : "max-w-[34rem]",
        )}
      >
        {children}
      </section>
    </motion.div>
  );
}
