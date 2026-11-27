import type { Variants } from "framer-motion";
import { duration, ease } from "@/components/styles/motion";

// Header shows first; the nav/chat items fade+rise in after a short delay,
// staggered. ease-out, well under Emil Kowalski's 300ms UI budget.
export const sidebarContainerVariants: Variants = {
  hidden: {},
  show: {
    transition: { delayChildren: 0.15, staggerChildren: 0.06 },
  },
};

export function getSidebarItemVariants(reduceMotion: boolean): Variants {
  if (reduceMotion) {
    return {
      hidden: { opacity: 0 },
      show: { opacity: 1, transition: { duration: duration.base } },
    };
  }

  return {
    hidden: { opacity: 0, y: 8, filter: "blur(4px)" },
    show: {
      opacity: 1,
      y: 0,
      filter: "blur(0px)",
      transition: { duration: duration.base, ease: ease.standard },
    },
  };
}
