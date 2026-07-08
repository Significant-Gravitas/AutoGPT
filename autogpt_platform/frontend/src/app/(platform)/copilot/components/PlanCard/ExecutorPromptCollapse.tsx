"use client";

import { CaretDownIcon, TerminalWindowIcon } from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useId, useState } from "react";
import { cn } from "@/lib/utils";

interface Props {
  prompt: string;
}

/**
 * Collapsed-by-default view of the exact `<execution_plan>` system-prompt
 * block handed to the executor. Lets the user inspect what each executor
 * turn was actually instructed to do.
 */
export function ExecutorPromptCollapse({ prompt }: Props) {
  const shouldReduceMotion = useReducedMotion();
  const contentId = useId();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="mt-3 rounded-lg border border-zinc-200 bg-white/60 px-3 py-2">
      <button
        type="button"
        aria-expanded={isOpen}
        aria-controls={contentId}
        onClick={() => setIsOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 text-left"
      >
        <span className="flex items-center gap-2 text-xs font-medium text-zinc-600">
          <TerminalWindowIcon className="size-4" />
          Prompt sent to executor
        </span>
        <CaretDownIcon
          className={cn(
            "size-3.5 shrink-0 text-zinc-400 transition-transform",
            isOpen && "rotate-180",
          )}
          weight="bold"
        />
      </button>

      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            id={contentId}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={
              shouldReduceMotion
                ? { duration: 0 }
                : { duration: 0.3, ease: [0.16, 1, 0.3, 1] }
            }
            className="overflow-hidden"
          >
            <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md bg-zinc-900 p-3 font-mono text-[11px] leading-relaxed text-zinc-100">
              {prompt}
            </pre>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
