"use client";

import { GlassOrb } from "@/components/molecules/GlassOrb/GlassOrb";
import { motion, useReducedMotion } from "framer-motion";
import {
  GREETING_ORB_LAYOUT_ID,
  ORB_FLIP_TRANSITION,
  SMALL_ORB_PARAMS,
} from "../../OnboardingIntroCard/OnboardingIntroCard";

// What the page is while the greeting is still being written: the orb,
// alone, breathing in the middle of the empty session. It carries the
// intro card's `layoutId`, so when the greeting lands this exact element
// travels into the heading instead of being swapped out for a copy.
export function GreetingLoader() {
  const prefersReducedMotion = useReducedMotion();

  return (
    <div
      role="status"
      className="flex w-full justify-center py-10"
      data-testid="greeting-loader"
    >
      {/* The orb is decorative and hidden from assistive tech, so the wait
          needs saying out loud — the composer is withheld until it ends. */}
      <span className="sr-only">Writing your greeting</span>
      <motion.span
        layoutId={GREETING_ORB_LAYOUT_ID}
        className="relative block size-16"
        // A fade rather than an instant paint: every copilot load renders
        // one pre-mount frame, and without this the flag-off user sees the
        // orb blink before the regular hero replaces it.
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{
          ...ORB_FLIP_TRANSITION,
          opacity: { duration: 0.25, delay: 0.15 },
        }}
      >
        <motion.span
          className="relative block size-full"
          // Draws in before it swells, so the beat reads as breathing
          // rather than a bar filling — a wait, not a progress step.
          animate={
            prefersReducedMotion ? undefined : { scale: [1, 0.85, 1.12, 1] }
          }
          transition={{
            duration: 2,
            times: [0, 0.3, 0.7, 1],
            repeat: Infinity,
            repeatDelay: 0.1,
            ease: "easeInOut",
          }}
        >
          <GlassOrb params={SMALL_ORB_PARAMS} />
        </motion.span>
      </motion.span>
    </div>
  );
}
