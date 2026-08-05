"use client";

import { domAnimation, LazyMotion, m, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import type { ChainRowState } from "./helpers";

interface Props {
  state: ChainRowState;
  label: string;
  morphToCheck: boolean;
  children: React.ReactNode;
}

const POP_EASE = [0.34, 1.35, 0.64, 1] as const;
const DRAW_EASE = [0.22, 1, 0.36, 1] as const;

export function ToolStatusBadge({
  state,
  label,
  morphToCheck,
  children,
}: Props) {
  const reducedMotion = useReducedMotion();
  const previousState = useRef(state);
  const [crossing, setCrossing] = useState(false);
  const done = state === "done" && morphToCheck;
  const running = state === "running";

  useEffect(() => {
    if (reducedMotion || previousState.current === state) return;
    previousState.current = state;
    setCrossing(true);
    const timer = window.setTimeout(() => setCrossing(false), 120);
    return () => window.clearTimeout(timer);
  }, [reducedMotion, state]);

  return (
    <LazyMotion features={domAnimation} strict>
      <m.span
        className="inline-flex"
        animate={{ filter: crossing ? "blur(0.5px)" : "blur(0px)" }}
        transition={{ duration: reducedMotion ? 0 : 0.12, ease: POP_EASE }}
      >
        <m.span
          data-state={done ? "done" : running ? "spinning" : state}
          role="img"
          aria-label={label}
          className="relative flex size-[22px] shrink-0 items-center justify-center"
          animate={
            done && !reducedMotion
              ? { scale: 1.09, y: [0, -3, 0] }
              : { scale: 1, y: 0 }
          }
          transition={
            done && !reducedMotion
              ? {
                  scale: { duration: 0.28, ease: POP_EASE },
                  y: { duration: 0.42, times: [0, 0.48, 1], ease: POP_EASE },
                }
              : { duration: reducedMotion ? 0 : 0.18, ease: DRAW_EASE }
          }
        >
          <m.span
            aria-hidden="true"
            className="absolute inset-0 rounded-full border-2 border-black/10"
            initial={{ opacity: running ? 1 : 0 }}
            animate={{ opacity: running ? 1 : 0 }}
            transition={{ duration: reducedMotion ? 0 : 0.16 }}
          />
          <m.span
            aria-hidden="true"
            className="absolute inset-0 animate-[spin_0.9s_linear_infinite] rounded-full border-2 border-transparent border-t-zinc-500 motion-reduce:animate-none"
            initial={{ opacity: running ? 1 : 0 }}
            animate={{ opacity: running ? 1 : 0 }}
            transition={{ duration: reducedMotion ? 0 : 0.12 }}
          />
          <m.span
            aria-hidden="true"
            className="absolute -inset-px rounded-full bg-[#35ba00] shadow-[inset_0_0_0_0.5px_rgba(0,0,0,0.05),0_0.5px_2px_rgba(0,0,0,0.12)]"
            initial={{ opacity: done ? 1 : 0 }}
            animate={{ opacity: done ? 1 : 0 }}
            transition={{ duration: reducedMotion ? 0 : 0.24, ease: POP_EASE }}
          />
          <m.span
            aria-hidden="true"
            className="relative flex items-center justify-center"
            initial={{ opacity: done ? 0 : 1, scale: done ? 0.9 : 1 }}
            animate={{ opacity: done ? 0 : 1, scale: done ? 0.9 : 1 }}
            transition={{ duration: reducedMotion ? 0 : 0.14, ease: DRAW_EASE }}
          >
            {children}
          </m.span>
          <svg
            aria-hidden="true"
            viewBox="0 0 24 24"
            className="absolute inset-0 size-full overflow-visible"
          >
            <m.path
              d="M8 12.5L10.8 15.5L16.4 9.5"
              fill="none"
              stroke="white"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              initial={{ pathLength: done ? 1 : 0, opacity: done ? 1 : 0 }}
              animate={{ pathLength: done ? 1 : 0, opacity: done ? 1 : 0 }}
              transition={{
                pathLength: {
                  duration: reducedMotion ? 0 : 0.32,
                  delay: done && !reducedMotion ? 0.1 : 0,
                  ease: DRAW_EASE,
                },
                opacity: { duration: reducedMotion ? 0 : 0.12 },
              }}
            />
          </svg>
        </m.span>
      </m.span>
    </LazyMotion>
  );
}
