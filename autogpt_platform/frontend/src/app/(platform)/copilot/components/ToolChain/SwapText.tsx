"use client";

import {
  AnimatePresence,
  domAnimation,
  LazyMotion,
  m,
  useReducedMotion,
} from "framer-motion";
import { useState } from "react";
import { ShimmerText } from "./ShimmerText";
import { useHeldText } from "./useHeldText";

const EASE_OUT_CUBIC = [0.33, 1, 0.68, 1] as const;
// A label that changes again while the previous swap could still be playing
// snaps instead of sliding — fast tools otherwise stack overlapping swap
// animations and the text jitters.
const RAPID_SWAP_MS = 250;

interface Swap {
  text: string;
  at: number;
  rapid: boolean;
}

function nextSwap(prev: Swap, text: string): Swap {
  const now = Date.now();
  return { text, at: now, rapid: now - prev.at < RAPID_SWAP_MS };
}

interface SwapTextProps {
  text: string;
  className?: string;
  shimmer?: boolean;
  /** Keep each label up for at least this long and then show the newest one,
   *  for labels that can change faster than they can be read. Off under
   *  reduced motion, where every change shows at once. */
  minHoldMs?: number;
}

const variants = {
  initial: (instant: boolean) =>
    instant ? { opacity: 1, y: 0 } : { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  exit: (instant: boolean) =>
    instant
      ? { opacity: 0, y: 0, transition: { duration: 0 } }
      : { opacity: 0, y: -8 },
};

// Old label slides up + fades out, new one slides in from below.
export function SwapText({
  text: latestText,
  className,
  shimmer = false,
  minHoldMs = 0,
}: SwapTextProps) {
  const reducedMotion = useReducedMotion();
  const text = useHeldText(latestText, reducedMotion ? 0 : minHoldMs);
  const [swap, setSwap] = useState<Swap>({ text, at: 0, rapid: false });
  let current = swap;
  if (swap.text !== text) {
    current = nextSwap(swap, text);
    setSwap(current);
  }
  const instant = !!reducedMotion || current.rapid;

  return (
    <LazyMotion features={domAnimation} strict>
      <span className={"inline-grid overflow-hidden " + (className ?? "")}>
        <AnimatePresence mode="popLayout" initial={false} custom={instant}>
          <m.span
            key={text}
            custom={instant}
            variants={variants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{
              duration: instant ? 0 : 0.18,
              ease: EASE_OUT_CUBIC,
            }}
            className="block truncate"
          >
            {shimmer ? <ShimmerText text={text} /> : text}
          </m.span>
        </AnimatePresence>
      </span>
    </LazyMotion>
  );
}
