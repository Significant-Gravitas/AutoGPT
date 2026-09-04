"use client";

import { cn } from "@/lib/utils";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { motion, useReducedMotion } from "framer-motion";
import { useMemo, useState } from "react";
import { tickColor, tickScale, toMinimapEntries } from "./helpers";

const TICK_SPRING = { type: "spring", stiffness: 200, damping: 15 } as const;
const CARD_TRANSITION = { duration: 0.15, delay: 0.0875 } as const;

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
}

/**
 * A tick rail down the chat's left gutter — one mark per message you sent.
 * Ticks sit shrunk until the cursor comes near, then swell toward it; hovering
 * one reveals that turn's opening line, and clicking scrolls the transcript to
 * it, so a long thread stays navigable without dragging the scrollbar.
 */
export function ChatMinimap({ messages }: Props) {
  const [hovered, setHovered] = useState<number | null>(null);
  const reducedMotion = useReducedMotion();
  const entries = useMemo(() => toMinimapEntries(messages), [messages]);

  if (entries.length < 3) return null;

  return (
    <div
      className="pointer-events-none absolute inset-y-0 left-0 z-20 hidden w-16 items-center xl:flex"
      onMouseLeave={() => setHovered(null)}
    >
      {/* No overflow clipping here — the hover card sits outside the rail's
          own box and would be cut off. */}
      <div className="pointer-events-auto flex max-h-full flex-col justify-center py-6 pl-3">
        {entries.map((entry, index) => (
          <div
            key={entry.id}
            className="relative cursor-pointer py-[3px]"
            onMouseEnter={() => setHovered(index)}
            onFocus={() => setHovered(index)}
            onBlur={() => setHovered(null)}
            onClick={() => scrollToMessage(entry.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                scrollToMessage(entry.id);
              }
              if (e.key === "Escape") {
                setHovered(null);
              }
            }}
            role="button"
            tabIndex={0}
            aria-label={`Jump to: ${entry.title}`}
          >
            <motion.div
              className={cn(
                "h-[4px] w-[32px] origin-left rounded-full transition-colors duration-150",
                tickColor(hovered === null ? null : Math.abs(index - hovered)),
              )}
              initial={{ scale: 0.6 }}
              animate={{ scale: tickScale(index, hovered) }}
              transition={reducedMotion ? { duration: 0 } : TICK_SPRING}
            />
            {hovered === index && (
              <motion.div
                initial={
                  reducedMotion
                    ? false
                    : { opacity: 0, scale: 0.4, filter: "blur(5px)" }
                }
                animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                transition={reducedMotion ? { duration: 0 } : CARD_TRANSITION}
                className="absolute left-[40px] top-1/2 z-30 w-80 origin-left -translate-y-1/2 rounded-2xl bg-white p-3.5 smooth-shadow-ring-sm"
              >
                <p className="truncate text-[15px] text-zinc-900">
                  {entry.title}
                </p>
                {entry.body && (
                  <p className="mt-1 line-clamp-3 text-[15px] leading-relaxed text-zinc-400">
                    {entry.body}
                  </p>
                )}
              </motion.div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function scrollToMessage(id: string) {
  document
    .querySelector(`[data-message-id="${CSS.escape(id)}"]`)
    ?.scrollIntoView({ behavior: "smooth", block: "center" });
}
