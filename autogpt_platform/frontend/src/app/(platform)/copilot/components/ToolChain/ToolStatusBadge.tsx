"use client";

import { cn } from "@/lib/utils";
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
        {/* -inset-[3px] lands the 1px ring exactly on the size-7 circle edge.
            Appearance is delayed so sub-150ms tools never flash the spinner;
            hiding is immediate. */}
        <m.span
          aria-hidden="true"
          className="absolute -inset-[3px] rounded-full border border-zinc-200"
          initial={{ opacity: 0 }}
          animate={{ opacity: running ? 1 : 0 }}
          transition={{
            duration: reducedMotion ? 0 : 0.16,
            delay: running && !reducedMotion ? 0.15 : 0,
          }}
        />
        <m.span
          aria-hidden="true"
          className={cn(
            "absolute -inset-[3px] rounded-full border border-transparent border-t-zinc-500",
            running &&
              "animate-[spin_0.9s_linear_infinite] motion-reduce:animate-none",
          )}
          initial={{ opacity: 0 }}
          animate={{ opacity: running ? 1 : 0 }}
          transition={{
            duration: reducedMotion ? 0 : 0.12,
            delay: running && !reducedMotion ? 0.15 : 0,
          }}
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
