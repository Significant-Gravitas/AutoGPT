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
import { duration, ease } from "@/components/styles/motion";

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
export function SwapText({ text, className, shimmer = false }: SwapTextProps) {
  const reducedMotion = useReducedMotion();
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
              duration: instant ? 0 : duration.base,
              ease: ease.out,
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
