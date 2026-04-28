"use client";

import { motion, useReducedMotion } from "framer-motion";
import type { Variants } from "framer-motion";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

const PLACEHOLDER_ROWS = Array.from({ length: 6 }, (_, i) => i);

const SKELETON_CONTAINER_VARIANTS: Variants = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.04, delayChildren: 0.04 },
  },
};

const SKELETON_ITEM_VARIANTS: Variants = {
  hidden: { opacity: 0, y: 6 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.28, ease: [0.16, 1, 0.3, 1] },
  },
};

export function APIKeyListSkeleton() {
  const reduceMotion = useReducedMotion();

  return (
    <motion.div
      role="status"
      aria-label="Loading API keys"
      className="flex w-full flex-col divide-y divide-zinc-200 overflow-hidden rounded-[8px] border border-zinc-200 bg-white"
      initial={reduceMotion ? false : "hidden"}
      animate={reduceMotion ? undefined : "show"}
      variants={reduceMotion ? undefined : SKELETON_CONTAINER_VARIANTS}
    >
      {PLACEHOLDER_ROWS.map((i) => (
        <motion.div
          key={i}
          className="flex items-center justify-between py-4 pl-3 pr-5"
          variants={reduceMotion ? undefined : SKELETON_ITEM_VARIANTS}
        >
          <div className="flex items-center gap-3">
            <Skeleton className="h-5 w-5" />
            <div className="flex flex-col gap-2">
              <Skeleton className="h-4 w-40" />
              <Skeleton className="h-3 w-56" />
            </div>
          </div>
          <Skeleton className="h-5 w-5" />
        </motion.div>
      ))}
    </motion.div>
  );
}
