import { useReducedMotion } from "framer-motion";
import { useEffect, useState } from "react";
import { BRAIN_DUMP_PROMPT_SETS, promptSetNameForRole } from "../prompts";

const SWAP_MS = 3_500;
const REDUCED_MOTION_SWAP_MS = 7_000;

export type AmbientSlot = "question" | "example";

interface Args {
  role: string;
  // What each card carries, in the order the cards are laid out.
  slots: AmbientSlot[];
  // Off wherever the layer is not on screen, and while the user has paused
  // the cards.
  isRotating: boolean;
}

export interface AmbientContent {
  label: string;
  text: string;
  isQuote: boolean;
}

export function useAmbientPrompts({ role, slots, isRotating }: Args) {
  const promptSet = promptSetNameForRole(role);
  const reduceMotion = useReducedMotion() === true;
  const [rotation, setRotation] = useState({ promptSet, swaps: 0 });
  const swaps = rotation.promptSet === promptSet ? rotation.swaps : 0;

  useEffect(() => {
    if (!isRotating) return;
    const timer = setInterval(
      () => {
        if (document.visibilityState !== "visible") return;
        setRotation((current) => ({
          promptSet,
          swaps: (current.promptSet === promptSet ? current.swaps : 0) + 1,
        }));
      },
      reduceMotion ? REDUCED_MOTION_SWAP_MS : SWAP_MS,
    );
    return () => clearInterval(timer);
  }, [promptSet, reduceMotion, isRotating]);

  const { questions, examples } = BRAIN_DUMP_PROMPT_SETS[promptSet];
  const dealt = dealCards(slots, swaps, {
    question: questions.length,
    example: examples.length,
  });
  const cards: AmbientContent[] = dealt.map((index, card) => {
    if (slots[card] === "example")
      return {
        label: "For example",
        text: `“${examples[index]}”`,
        isQuote: true,
      };
    return { ...questions[index], isQuote: false };
  });
  return { cards, reduceMotion };
}

// Which card the next swap lands on, for the six-card layout: it hops
// across the page rather than walking down one side, so each change pulls
// the eye somewhere new.
const SWAP_ORDER = [3, 1, 5, 0, 4, 2];

function swapOrder(cardCount: number) {
  if (cardCount === SWAP_ORDER.length) return SWAP_ORDER;
  return Array.from({ length: cardCount }, (_, card) => card);
}

// Deals the opening hand (questions and quotes in order), then replays every
// swap: the swapped card takes the next item of its own kind nobody has seen
// yet, wrapping round once the set runs out.
export function dealCards(
  slots: AmbientSlot[],
  swaps: number,
  sizes: Record<AmbientSlot, number>,
) {
  const next: Record<AmbientSlot, number> = { question: 0, example: 0 };
  const shown = slots.map((slot) => next[slot]++ % sizes[slot]);
  const order = swapOrder(slots.length);
  for (let swap = 0; swap < swaps; swap++) {
    const card = order[swap % order.length];
    const slot = slots[card];
    shown[card] = next[slot]++ % sizes[slot];
  }
  return shown;
}
