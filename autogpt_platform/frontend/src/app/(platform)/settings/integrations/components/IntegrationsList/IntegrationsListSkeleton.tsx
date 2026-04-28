"use client";

import { motion, useReducedMotion } from "framer-motion";
import type { Variants } from "framer-motion";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

const SKELETON_CONTAINER_VARIANTS: Variants = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.08, delayChildren: 0.05 },
  },
};

const SKELETON_ITEM_VARIANTS: Variants = {
  hidden: { opacity: 0, y: 16 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.3, ease: [0.16, 1, 0.3, 1] },
  },
};

const REDUCED_MOTION_ITEM_VARIANTS: Variants = {
  hidden: { opacity: 0 },
  show: { opacity: 1 },
};

export function IntegrationsListSkeleton() {
  const reduceMotion = useReducedMotion();

  return (
    <motion.div
      className="flex flex-col gap-3"
      aria-busy="true"
      aria-label="Loading integrations"
      initial={reduceMotion ? false : "hidden"}
      animate={reduceMotion ? undefined : "show"}
      variants={reduceMotion ? undefined : SKELETON_CONTAINER_VARIANTS}
    >
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          data-testid="integration-skeleton-item"
          className="w-full overflow-hidden rounded-lg border border-[#DADADC] bg-white"
          variants={
            reduceMotion ? REDUCED_MOTION_ITEM_VARIANTS : SKELETON_ITEM_VARIANTS
          }
        >
          {/* Mirrors ProviderGroup accordion trigger row */}
          <div className="flex items-center justify-between px-3 py-3 pr-5">
            <div className="flex items-center gap-3">
              <Skeleton className="size-6 rounded-full" />
              <Skeleton className="h-[22px] w-32" />
              <Skeleton className="h-[22px] w-8 rounded-[10px]" />
            </div>
            <Skeleton className="size-4 rounded" />
          </div>
          {/* Mirrors first credential row inside accordion content */}
          <div className="flex items-center justify-between border-t border-[#DADADC] py-3 pl-3 pr-5">
            <div className="flex items-center gap-3">
              <Skeleton className="size-5 rounded" />
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center gap-3">
                  <Skeleton className="h-[22px] w-40" />
                  <Skeleton className="h-[20px] w-14 rounded-[10px]" />
                </div>
                <Skeleton className="h-3 w-28" />
              </div>
            </div>
            <Skeleton className="size-5 rounded" />
          </div>
        </motion.div>
      ))}
    </motion.div>
  );
}
