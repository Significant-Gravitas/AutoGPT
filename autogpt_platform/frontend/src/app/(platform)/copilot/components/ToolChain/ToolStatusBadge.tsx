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
            The ring shows the moment a call starts: a fast call holds its
            running look for a minimum time instead (useMinimumRunning), so
            it reads as having run rather than popping in finished. */}
        <m.span
          aria-hidden="true"
          className="absolute -inset-[3px] rounded-full border border-border-base"
          initial={{ opacity: running ? 1 : 0 }}
          animate={{ opacity: running ? 1 : 0 }}
          transition={{ duration: reducedMotion ? 0 : 0.12, ease: "easeOut" }}
        />
        <m.span
          aria-hidden="true"
          className={cn(
            "absolute -inset-[3px] rounded-full border border-transparent border-t-foreground-tertiary",
            running && "animate-spin motion-reduce:animate-none",
          )}
          initial={{ opacity: running ? 1 : 0 }}
          animate={{ opacity: running ? 1 : 0 }}
          transition={{ duration: reducedMotion ? 0 : 0.12, ease: "easeOut" }}
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
