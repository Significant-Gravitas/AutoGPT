"use client";

import { domAnimation, LazyMotion, m, useReducedMotion } from "framer-motion";
import type { ChainRowState } from "./helpers";

interface Props {
  state: ChainRowState;
  label: string;
  children: React.ReactNode;
}

export function ToolStatusBadge({ state, label, children }: Props) {
  const reducedMotion = useReducedMotion();
  const running = state === "running";

  return (
    <LazyMotion features={domAnimation} strict>
      <m.span
        data-state={running ? "spinning" : state}
        role="img"
        aria-label={label}
        className="relative flex size-[22px] shrink-0 items-center justify-center"
      >
        {/* -inset-[3px] lands the 1px ring exactly on the size-7 circle edge. */}
        <m.span
          aria-hidden="true"
          className="absolute -inset-[3px] rounded-full border border-zinc-200"
          initial={{ opacity: running ? 1 : 0 }}
          animate={{ opacity: running ? 1 : 0 }}
          transition={{ duration: reducedMotion ? 0 : 0.16 }}
        />
        <m.span
          aria-hidden="true"
          className="absolute -inset-[3px] animate-[spin_0.9s_linear_infinite] rounded-full border border-transparent border-t-zinc-500 motion-reduce:animate-none"
          initial={{ opacity: running ? 1 : 0 }}
          animate={{ opacity: running ? 1 : 0 }}
          transition={{ duration: reducedMotion ? 0 : 0.12 }}
        />
        <span
          aria-hidden="true"
          className="relative flex items-center justify-center"
        >
          {children}
        </span>
      </m.span>
    </LazyMotion>
  );
}
