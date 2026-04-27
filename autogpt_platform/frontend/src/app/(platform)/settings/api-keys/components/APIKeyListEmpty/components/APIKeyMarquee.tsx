"use client";

import { motion, useReducedMotion } from "framer-motion";
import { KeyIcon } from "@phosphor-icons/react";

const GHOST_CARDS = Array.from({ length: 4 }, (_, i) => i);

export function APIKeyMarquee() {
  const reduceMotion = useReducedMotion();

  return (
    <div
      aria-hidden
      className="relative h-[260px] w-[340px] overflow-hidden"
      style={{
        maskImage:
          "linear-gradient(to bottom, transparent 0%, black 35%, black 65%, transparent 100%)",
        WebkitMaskImage:
          "linear-gradient(to bottom, transparent 0%, black 35%, black 65%, transparent 100%)",
      }}
    >
      <motion.div
        className="flex flex-col items-center gap-4 will-change-transform"
        animate={reduceMotion ? undefined : { y: ["0%", "-50%"] }}
        transition={
          reduceMotion
            ? undefined
            : { duration: 14, ease: "linear", repeat: Infinity }
        }
      >
        {[...GHOST_CARDS, ...GHOST_CARDS].map((_, i) => (
          <GhostCard key={i} />
        ))}
      </motion.div>
    </div>
  );
}

function GhostCard() {
  return (
    <div className="flex h-[64px] w-[320px] items-center gap-4 rounded-xl border border-zinc-200/80 bg-white px-5 shadow-[0_1px_2px_rgba(0,0,0,0.04)]">
      <KeyIcon size={18} className="shrink-0 text-zinc-400" />
      <div className="h-2.5 flex-1 rounded-full bg-zinc-100" />
    </div>
  );
}
