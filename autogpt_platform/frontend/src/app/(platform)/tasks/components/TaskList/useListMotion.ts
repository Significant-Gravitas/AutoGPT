import { useReducedMotion } from "framer-motion";

const GLIDE = { duration: 0.4, ease: [0.22, 1, 0.36, 1] } as const;

/** Rows rise in, glide to their new place and fade out; nothing moves
 *  under reduced motion. */
export function useListMotion(layoutID?: string) {
  const reduceMotion = useReducedMotion();

  if (reduceMotion) {
    return {
      layout: false,
      initial: false,
      transition: { duration: 0 },
    } as const;
  }

  return {
    layout: "position",
    layoutId: layoutID,
    initial: { opacity: 0, y: 6 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, transition: { duration: 0.16 } },
    transition: GLIDE,
  } as const;
}
