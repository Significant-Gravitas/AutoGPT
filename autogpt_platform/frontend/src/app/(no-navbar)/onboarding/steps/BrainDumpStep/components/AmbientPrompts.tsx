"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "framer-motion";
import type { ScreenState } from "../useBrainDumpStep";
import { useAmbientPrompts } from "./useAmbientPrompts";

interface Props {
  role: string;
  screen: ScreenState;
}

type Card = { position: string; driftSecs: number; delaySecs: number } & (
  | { question: number; besideComposer?: boolean }
  | { example: number }
);

// Left column then right. The middle pair sits beside the composer, so it
// steps out while the user types.
const CARDS: Card[] = [
  { position: "left-[3%] top-[5%]", driftSecs: 9, delaySecs: 0, question: 0 },
  {
    position: "left-[7%] top-[36%]",
    driftSecs: 11,
    delaySecs: 1.5,
    question: 1,
    besideComposer: true,
  },
  { position: "left-[4%] top-[66%]", driftSecs: 13, delaySecs: 3, example: 0 },
  { position: "right-[3%] top-[7%]", driftSecs: 11, delaySecs: 2, question: 2 },
  {
    position: "right-[6%] top-[38%]",
    driftSecs: 13,
    delaySecs: 0.5,
    question: 3,
    besideComposer: true,
  },
  {
    position: "right-[4%] top-[68%]",
    driftSecs: 9,
    delaySecs: 2.5,
    example: 1,
  },
];

const HIDDEN_ON: ScreenState[] = ["recovery", "failed", "insufficient"];

// Starter ideas around the column on wide screens (see useIsWideLayout). Text, not controls: no
// border, shadow, hover or focus, and hidden from assistive tech, so that
// nobody mistakes them for buttons. The live prompt is the accessible one.
export function AmbientPrompts({ role, screen }: Props) {
  const ambient = useAmbientPrompts(role);
  if (HIDDEN_ON.includes(screen)) return null;
  const isTyping = screen === "typing";

  return (
    <div
      aria-hidden="true"
      data-testid="ambient-prompts"
      className={cn(
        "pointer-events-none absolute inset-0 z-0 hidden select-none transition-opacity duration-500 xl:block",
        (isTyping || screen === "recording") && "opacity-50",
      )}
    >
      {CARDS.map((card) => {
        const isQuote = "example" in card;
        const text = isQuote
          ? `“${ambient.examples[card.example]}”`
          : ambient.questions[card.question].text;
        return (
          <motion.div
            key={card.position}
            className={cn(
              "absolute w-[196px] rounded-xl bg-surface-hover px-3 py-2.5 transition-opacity duration-500",
              card.position,
              isTyping && "besideComposer" in card && "opacity-0",
            )}
            animate={ambient.reduceMotion ? undefined : { y: [0, -6, 0] }}
            transition={
              ambient.reduceMotion
                ? undefined
                : {
                    duration: card.driftSecs,
                    delay: card.delaySecs,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }
            }
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={text}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: ambient.reduceMotion ? 0 : 0.4 }}
                className={cn(isQuote && "italic")}
              >
                <Text
                  variant="small-semibold"
                  as="span"
                  className="block text-caption-sm uppercase tracking-eyebrow text-accent-strong"
                >
                  {isQuote
                    ? "For example"
                    : ambient.questions[card.question].label}
                </Text>
                <Text
                  variant="body"
                  as="span"
                  tone="secondary"
                  className="block text-body-sm"
                >
                  {text}
                </Text>
              </motion.div>
            </AnimatePresence>
          </motion.div>
        );
      })}
    </div>
  );
}
