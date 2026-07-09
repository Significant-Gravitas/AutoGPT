"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";

interface Props {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
}

/**
 * Popover presentation of the workspace files card, used while the sandbox IDE
 * occupies the right side — there's no room for the inline panel, so the card
 * floats over the chat, anchored to the header toggle at the top-right of the
 * inset. The sandbox is a fixed 50vw panel, so the right offset is 50vw + gap.
 */
export function ContextPanelPopover({ open, onClose, children }: Props) {
  const reduceMotion = useReducedMotion();

  return (
    <AnimatePresence>
      {open && (
        <>
          <div className="fixed inset-0 z-40" aria-hidden onClick={onClose} />
          <motion.div
            role="dialog"
            initial={
              reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.96, y: -8 }
            }
            animate={
              reduceMotion ? { opacity: 1 } : { opacity: 1, scale: 1, y: 0 }
            }
            exit={
              reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.96, y: -8 }
            }
            transition={
              reduceMotion
                ? { duration: 0.12 }
                : { duration: 0.2, ease: [0.32, 0.72, 0, 1] }
            }
            style={{ willChange: "transform" }}
            className="fixed right-[calc(50vw+1rem)] top-[4.5rem] z-50 flex max-h-[70vh] w-[22rem] origin-top-right flex-col overflow-hidden rounded-[2rem] border border-zinc-100 bg-white shadow-xl [corner-shape:squircle]"
          >
            {children}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
