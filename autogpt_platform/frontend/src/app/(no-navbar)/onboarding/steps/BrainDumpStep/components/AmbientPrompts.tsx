"use client";

import { cn } from "@/lib/utils";
import type { ScreenState } from "../useBrainDumpStep";
import { useIsWideLayout } from "../useIsWideLayout";
import { AmbientCard } from "./AmbientCard";
import { type AmbientSlot, useAmbientPrompts } from "./useAmbientPrompts";

interface Props {
  role: string;
  screen: ScreenState;
  isPaused: boolean;
  onTogglePause: () => void;
}

interface Card {
  position: string;
  driftSecs: number;
  delaySecs: number;
  slot: AmbientSlot;
  besideComposer?: boolean;
}

// Left column then right. The middle pair sits beside the composer, so it
// steps out while the user types.
const CARDS: Card[] = [
  {
    position: "left-[3%] top-[5%]",
    driftSecs: 7.5,
    delaySecs: 0,
    slot: "question",
  },
  {
    position: "left-[7%] top-[36%]",
    driftSecs: 9,
    delaySecs: 1.5,
    slot: "question",
    besideComposer: true,
  },
  {
    position: "left-[4%] top-[66%]",
    driftSecs: 8.2,
    delaySecs: 3,
    slot: "example",
  },
  {
    position: "right-[3%] top-[7%]",
    driftSecs: 8.8,
    delaySecs: 2,
    slot: "question",
  },
  {
    position: "right-[6%] top-[38%]",
    driftSecs: 7.2,
    delaySecs: 0.5,
    slot: "question",
    besideComposer: true,
  },
  {
    position: "right-[4%] top-[68%]",
    driftSecs: 9.6,
    delaySecs: 2.5,
    slot: "example",
  },
];

const SLOTS = CARDS.map((card) => card.slot);
const HIDDEN_ON: ScreenState[] = ["recovery", "failed", "insufficient"];

// Starter ideas around the column on wide screens (see useIsWideLayout). Text,
// not controls: no border, shadow, hover or focus, and hidden from assistive
// tech; the live prompt is the accessible one. They keep changing while the
// user talks, for inspiration mid-thought, and a click on any card pauses or
// resumes them all (the keyboard route is the focus-only toggle beside the
// layer, so WCAG 2.2.2 holds without a visible control).
export function AmbientPrompts({
  role,
  screen,
  isPaused,
  onTogglePause,
}: Props) {
  const isShown = useIsWideLayout() && !HIDDEN_ON.includes(screen);
  // Held still while the dump is being sent: there is no pause toggle on
  // that screen, and nothing left to inspire.
  const isStill = isPaused || screen === "processing";
  const ambient = useAmbientPrompts({
    role,
    slots: SLOTS,
    isRotating: isShown && !isStill,
  });
  if (HIDDEN_ON.includes(screen)) return null;
  const isTyping = screen === "typing";

  return (
    <div
      aria-hidden="true"
      data-testid="ambient-prompts"
      data-paused={isPaused}
      className={cn(
        "pointer-events-none absolute inset-0 z-0 hidden select-none transition-opacity duration-500 xl:block",
        (isTyping || screen === "recording") && "opacity-50",
      )}
    >
      {CARDS.map((card, index) => {
        const isHidden = isTyping && Boolean(card.besideComposer);
        return (
          <AmbientCard
            key={card.position}
            position={card.position}
            content={ambient.cards[index]}
            isHidden={isHidden}
            drift={
              isShown && !isStill && !ambient.reduceMotion
                ? { durationSecs: card.driftSecs, delaySecs: card.delaySecs }
                : null
            }
            reduceMotion={ambient.reduceMotion}
            onClick={isHidden ? undefined : onTogglePause}
          />
        );
      })}
    </div>
  );
}
