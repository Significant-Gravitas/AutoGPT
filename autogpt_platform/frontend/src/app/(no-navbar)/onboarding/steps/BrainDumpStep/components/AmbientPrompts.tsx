"use client";

import { cn } from "@/lib/utils";
import type { ScreenState } from "../useBrainDumpStep";
import { useIsWideLayout } from "../useIsWideLayout";
import { AmbientCard } from "./AmbientCard";
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
  const isShown = useIsWideLayout() && !HIDDEN_ON.includes(screen);
  const ambient = useAmbientPrompts({
    role,
    isRotating: isShown && screen !== "recording",
  });
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
        return (
          <AmbientCard
            key={card.position}
            position={card.position}
            label={
              isQuote ? "For example" : ambient.questions[card.question].label
            }
            text={
              isQuote
                ? `“${ambient.examples[card.example]}”`
                : ambient.questions[card.question].text
            }
            isQuote={isQuote}
            isHidden={isTyping && "besideComposer" in card}
            drift={
              isShown && !ambient.reduceMotion
                ? { durationSecs: card.driftSecs, delaySecs: card.delaySecs }
                : null
            }
            reduceMotion={ambient.reduceMotion}
          />
        );
      })}
    </div>
  );
}
