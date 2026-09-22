"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { formatElapsed } from "../helpers";

// Each digit is its own slot, so only the digits that actually changed pop:
// the seconds tick every second while the minutes sit still.
export function ElapsedTime({ seconds }: { seconds: number }) {
  const prefersReducedMotion = useReducedMotion();
  const characters = formatElapsed(seconds).split("");

  return (
    <span
      className="flex items-center font-poppins text-[2rem] font-medium tabular-nums leading-[2.5rem] text-zinc-900"
      aria-label={`${seconds} seconds recorded`}
      role="timer"
    >
      {characters.map((character, index) => (
        <span
          key={index}
          className="relative inline-flex justify-center"
          style={{ width: character === ":" ? "0.3em" : "0.62em" }}
          aria-hidden
        >
          <AnimatePresence initial={false} mode="popLayout">
            <motion.span
              key={character}
              initial={
                prefersReducedMotion
                  ? { opacity: 0 }
                  : { opacity: 0, scale: 0.6, y: 8 }
              }
              animate={
                prefersReducedMotion
                  ? { opacity: 1 }
                  : { opacity: 1, scale: 1, y: 0 }
              }
              exit={
                prefersReducedMotion
                  ? { opacity: 0 }
                  : { opacity: 0, scale: 0.6, y: -8 }
              }
              transition={{
                type: "spring",
                stiffness: 520,
                damping: 28,
                mass: 0.6,
              }}
            >
              {character}
            </motion.span>
          </AnimatePresence>
        </span>
      ))}
    </span>
  );
}
