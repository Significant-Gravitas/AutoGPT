"use client";

import {
  AnimatePresence,
  domAnimation,
  LazyMotion,
  m,
  useReducedMotion,
} from "framer-motion";
import { ShimmerText } from "./ShimmerText";

const EASE_OUT_CUBIC = [0.33, 1, 0.68, 1] as const;

interface SwapTextProps {
  text: string;
  className?: string;
  shimmer?: boolean;
}

// Old label slides up + fades out, new one slides in from below.
export function SwapText({ text, className, shimmer = false }: SwapTextProps) {
  const reducedMotion = useReducedMotion();
  return (
    <LazyMotion features={domAnimation} strict>
      <span className={"inline-grid overflow-hidden " + (className ?? "")}>
        <AnimatePresence mode="popLayout" initial={false}>
          <m.span
            key={text}
            initial={reducedMotion ? false : { opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reducedMotion ? undefined : { opacity: 0, y: -8 }}
            transition={{ duration: 0.18, ease: EASE_OUT_CUBIC }}
            className="block truncate"
          >
            {shimmer ? <ShimmerText text={text} /> : text}
          </m.span>
        </AnimatePresence>
      </span>
    </LazyMotion>
  );
}
