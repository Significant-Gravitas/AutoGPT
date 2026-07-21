import type { Variants } from "framer-motion";

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
      show: { opacity: 1, transition: { duration: 0.2 } },
    };
  }

  return {
    hidden: { opacity: 0, y: 8, filter: "blur(4px)" },
    show: {
      opacity: 1,
      y: 0,
      filter: "blur(0px)",
      transition: { duration: 0.25, ease: [0.25, 0.1, 0.25, 1] },
    },
  };
}
